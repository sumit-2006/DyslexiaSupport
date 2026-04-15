import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertNoteSchema } from "@shared/schema";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";
import multer from "multer";
import path from "path";
import fs from "fs";
import { createWorker } from "tesseract.js";
import OpenAI from "openai";

import ocrRoutes from "./routes/ocrRoutes";

// Initialize OpenAI (optional – only used if OPENAI_API_KEY is set)
const openai = process.env.OPENAI_API_KEY
  ? new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
  : null;

export async function registerRoutes(app: Express): Promise<Server> {
  // Register OCR routes
  app.use('/api/ocr', ocrRoutes);

  // ── Notes CRUD ──────────────────────────────────────────────────────────────

  // Get all notes
  app.get("/api/notes", async (req, res) => {
    try {
      const notes = await storage.getAllNotes();
      res.json(notes);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch notes" });
    }
  });

  // Get a single note
  app.get("/api/notes/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const note = await storage.getNote(id);
      if (!note) {
        return res.status(404).json({ message: "Note not found" });
      }
      res.json(note);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch note" });
    }
  });

  // Create a new note
  app.post("/api/notes", async (req, res) => {
    try {
      const parsed = insertNoteSchema.safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ message: fromZodError(parsed.error).message });
      }
      const note = await storage.createNote(parsed.data);
      res.status(201).json(note);
    } catch (error) {
      if (error instanceof ZodError) {
        return res.status(400).json({ message: fromZodError(error).message });
      }
      res.status(500).json({ message: "Failed to create note" });
    }
  });

  // Update a note
  app.put("/api/notes/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const parsed = insertNoteSchema.partial().safeParse(req.body);
      if (!parsed.success) {
        return res.status(400).json({ message: fromZodError(parsed.error).message });
      }
      const note = await storage.updateNote(id, parsed.data);
      if (!note) {
        return res.status(404).json({ message: "Note not found" });
      }
      res.json(note);
    } catch (error) {
      res.status(500).json({ message: "Failed to update note" });
    }
  });

  // Toggle favorite
  app.patch("/api/notes/:id/favorite", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const { isFavorite } = req.body;
      const note = await storage.updateNote(id, { isFavorite });
      if (!note) {
        return res.status(404).json({ message: "Note not found" });
      }
      res.json(note);
    } catch (error) {
      res.status(500).json({ message: "Failed to update favorite status" });
    }
  });

  // Delete a note
  app.delete("/api/notes/:id", async (req, res) => {
    try {
      const id = parseInt(req.params.id);
      const success = await storage.deleteNote(id);
      if (!success) {
        return res.status(404).json({ message: "Note not found" });
      }
      res.status(204).send();
    } catch (error) {
      res.status(500).json({ message: "Failed to delete note" });
    }
  });

  // ── OCR / Text Recognition ──────────────────────────────────────────────────

  const rawDir = path.join("uploads", "raw");
  if (!fs.existsSync(rawDir)) {
    fs.mkdirSync(rawDir, { recursive: true });
  }

  const storageEngine = multer.diskStorage({
    destination: (req, file, cb) => cb(null, rawDir),
    filename: (req, file, cb) => {
      const uniqueName = `${Date.now()}${path.extname(file.originalname)}`;
      cb(null, uniqueName);
    },
  });

  const upload = multer({
    storage: storageEngine,
    limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  });

  // Recognize text using Tesseract.js on the server
  app.post("/api/recognize-text", upload.single("image"), async (req, res) => {
    const filePath = req.file?.path;
    if (!filePath) {
      return res.status(400).json({ message: "No image uploaded" });
    }

    let worker: Awaited<ReturnType<typeof createWorker>> | null = null;
    try {
      worker = await createWorker("eng", 1, {
        logger: () => {}, // silence progress logs
      });

      // Tesseract PSM 6 = assume a single uniform block of text
      await worker.setParameters({ tessedit_pageseg_mode: "6" as any });

      const { data } = await worker.recognize(filePath);
      const rawText = data.text.trim();

      // Build dyslexia-specific correction suggestions
      const suggestions = buildSuggestions(rawText);

      // Formatted text: clean up excessive whitespace
      const formattedText = rawText
        .replace(/\n{3,}/g, "\n\n")
        .replace(/[ \t]{2,}/g, " ")
        .trim();

      res.status(200).json({ text: rawText, formattedText, suggestions });
    } catch (err: any) {
      console.error("[OCR] Error recognizing text:", err);
      res.status(500).json({ message: "OCR failed", error: err.message });
    } finally {
      if (worker) {
        try { await worker.terminate(); } catch (_) {}
      }
      // Clean up uploaded file
      try { fs.unlinkSync(filePath); } catch (_) {}
    }
  });

  // ── AI Correction ───────────────────────────────────────────────────────────

  app.post("/api/ai-correct", async (req, res) => {
    const { text } = req.body;
    if (!text || typeof text !== "string") {
      return res.status(400).json({ message: "text field is required" });
    }

    if (!openai) {
      // Fallback: just return the same text with suggestions
      return res.json({
        correctedText: text,
        suggestions: buildSuggestions(text),
      });
    }

    try {
      const completion = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are a helpful assistant for people with dyslexia. " +
              "Correct the following text: fix spelling, grammar, and common " +
              "dyslexia-related errors (b/d reversals, p/q, letter omissions, " +
              "word order issues). Return ONLY the corrected text, no explanation.",
          },
          { role: "user", content: text },
        ],
        max_tokens: 1024,
        temperature: 0.2,
      });

      const correctedText = completion.choices[0]?.message?.content?.trim() ?? text;
      res.json({ correctedText, suggestions: buildSuggestions(correctedText) });
    } catch (err: any) {
      console.error("[AI] OpenAI correction failed:", err.message);
      // Graceful fallback
      res.json({ correctedText: text, suggestions: buildSuggestions(text) });
    }
  });

  // ── Stats ───────────────────────────────────────────────────────────────────

  app.get("/api/stats", async (req, res) => {
    try {
      const notes = await storage.getAllNotes();
      const totalNotes = notes.length;
      const favoriteNotes = notes.filter((n) => n.isFavorite).length;

      // Count total words in recognized text
      const totalWords = notes.reduce((sum, note) => {
        if (!note.recognizedText) return sum;
        return sum + note.recognizedText.split(/\s+/).filter(Boolean).length;
      }, 0);

      // Recent activity (last 7 days)
      const sevenDaysAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
      const recentNotes = notes.filter(
        (n) => n.createdAt && new Date(n.createdAt) > sevenDaysAgo,
      ).length;

      res.json({ totalNotes, favoriteNotes, totalWords, recentNotes });
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch stats" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Generate dyslexia-aware correction suggestions for common letter reversals
 * and misspellings that Tesseract might produce.
 */
function buildSuggestions(
  text: string,
): Array<{ original: string; correction: string }> {
  const suggestions: Array<{ original: string; correction: string }> = [];

  // Common dyslexia OCR confusions
  const reversalPairs: [RegExp, string][] = [
    [/\bwas\b/g, "was"],   // 'saw' vs 'was' — keep as hint
    [/\bsaw\b/g, "saw"],
    [/\bon\b/g, "on"],
    [/\bno\b/g, "no"],
  ];

  // Find simple common OCR errors
  const commonErrors: Record<string, string> = {
    teh: "the",
    adn: "and",
    taht: "that",
    hte: "the",
    waht: "what",
    tihs: "this",
    yuo: "you",
    youre: "you're",
    dont: "don't",
    cant: "can't",
    wont: "won't",
    iam: "I am",
    im: "I'm",
  };

  const words = text.split(/\s+/);
  const seen = new Set<string>();

  for (const word of words) {
    const lower = word.toLowerCase().replace(/[^a-z']/g, "");
    if (lower && !seen.has(lower) && commonErrors[lower]) {
      seen.add(lower);
      suggestions.push({ original: word, correction: commonErrors[lower] });
    }
  }

  return suggestions;
}
