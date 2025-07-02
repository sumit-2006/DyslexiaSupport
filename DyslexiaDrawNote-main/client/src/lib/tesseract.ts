import { createWorker } from 'tesseract.js';

interface RecognitionResult {
  text: string;
  suggestions: Array<{
    original: string;
    correction: string;
  }>;
  formattedText: string; // Added for improved font styling
}

// Pre-process the image to improve recognition
async function preprocessImage(imageData: string): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    
    img.onload = () => {
      // Create an offscreen canvas for image processing
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(imageData);
        return;
      }
      
      // Draw the original image
      ctx.drawImage(img, 0, 0);
      
      // Get image data for processing
      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;
      
      // Apply image enhancements for better recognition
      for (let i = 0; i < data.length; i += 4) {
        // Convert to grayscale with improved contrast
        const brightness = 0.34 * data[i] + 0.5 * data[i + 1] + 0.16 * data[i + 2];
        
        // Apply thresholding to make text stand out more
        const threshold = 180;
        const value = brightness < threshold ? 0 : 255;
        
        // Set RGB values to the new value
        data[i] = value;     // Red
        data[i + 1] = value; // Green
        data[i + 2] = value; // Blue
        // Alpha channel remains unchanged
      }
      
      // Put processed image data back to canvas
      ctx.putImageData(imgData, 0, 0);
      
      // Return processed image as data URL
      resolve(canvas.toDataURL());
    };
    
    img.src = imageData;
  });
}

// Format recognized text to standardized computer fonts
function formatToStandardFont(text: string): string {
  // Format the text with appropriate line breaks and spacing
  return text
    .replace(/\n{3,}/g, '\n\n') // Replace multiple line breaks with double line break
    .replace(/\s{2,}/g, ' ')    // Replace multiple spaces with single space
    .trim();                    // Remove leading and trailing whitespace
}

export async function recognizeText(
  imageData: string
): Promise<RecognitionResult> {
  try {
    // Preprocess the image for better recognition
    const processedImage = await preprocessImage(imageData);
    
    // Configure Tesseract with better settings for handwriting
    const worker = await createWorker('eng');
    
    // Recognize text from the processed image
    const result = await worker.recognize(processedImage);
    await worker.terminate();

    // Get the recognized text
    const text = result.data.text;
    
    // Format text to look like computer font
    const formattedText = formatToStandardFont(text);
    
    // Check for spelling errors
    const words = text.split(/\s+/);
    const suggestions: Array<{ original: string; correction: string }> = [];

    // Enhanced list of common misspellings, especially for dyslexic users
    const commonMisspellings: Record<string, string> = {
      // Original list
      developement: "development",
      progres: "progress",
      recieve: "receive",
      freind: "friend",
      thier: "their",
      speling: "spelling",
      occured: "occurred",
      concieve: "conceive",
      wierd: "weird",
      acheive: "achieve",
      seperate: "separate",
      definately: "definitely",
      accomodate: "accommodate",
      untill: "until",
      beleive: "believe",
      existance: "existence",
      goverment: "government",
      enviroment: "environment",
      occassion: "occasion",
      dissapear: "disappear",
      tommorow: "tomorrow",
      begining: "beginning",
      comming: "coming",
      independant: "independent",
      successfull: "successful",
      
      // Additional dyslexia-specific misspellings
      becuase: "because",
      beutiful: "beautiful",
      diferent: "different",
      frist: "first",
      peice: "piece",
      peopel: "people",
      rember: "remember",
      togehter: "together",
      wich: "which",
      writting: "writing",
      acess: "access",
      alot: "a lot",
      alway: "always",
      anwser: "answer",
      befor: "before",
      comand: "command",
      grammer: "grammar",
      litrally: "literally",
      mispell: "misspell",
      prefrence: "preference",
      probabilty: "probability"
    };

    words.forEach(word => {
      const cleanWord = word.toLowerCase().replace(/[^\w]/g, '');
      if (commonMisspellings[cleanWord]) {
        suggestions.push({
          original: cleanWord,
          correction: commonMisspellings[cleanWord]
        });
      }
    });

    return {
      text,
      suggestions,
      formattedText
    };
  } catch (error) {
    console.error('Text recognition error:', error);
    return {
      text: '',
      suggestions: [],
      formattedText: ''
    };
  }
}
