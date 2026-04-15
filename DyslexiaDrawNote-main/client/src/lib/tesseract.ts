interface RecognitionResult {
  text: string;
  suggestions: Array<{
    original: string;
    correction: string;
  }>;
  formattedText: string;
}

/**
 * Send the canvas image to the server for OCR via Tesseract.js.
 */
export async function recognizeText(imageDataUrl: string): Promise<RecognitionResult> {
  const blob = await (await fetch(imageDataUrl)).blob();
  const formData = new FormData();
  formData.append("image", blob, "input.png");

  const response = await fetch("/api/recognize-text", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Recognition failed");
  }

  const data = await response.json();

  return {
    text: data.text ?? "",
    formattedText: data.formattedText ?? data.text ?? "",
    suggestions: Array.isArray(data.suggestions) ? data.suggestions : [],
  };
}

/**
 * Ask the server (OpenAI) to intelligently correct dyslexia-related errors.
 */
export async function aiCorrectText(text: string): Promise<{
  correctedText: string;
  suggestions: Array<{ original: string; correction: string }>;
}> {
  const response = await fetch("/api/ai-correct", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    throw new Error("AI correction failed");
  }

  return response.json();
}

