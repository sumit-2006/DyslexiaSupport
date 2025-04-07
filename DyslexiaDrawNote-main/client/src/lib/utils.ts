import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: Date | string): string {
  if (typeof date === 'string') {
    date = new Date(date);
  }
  
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

export function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength) + '...';
}

export function getCanvasPreview(canvas: HTMLCanvasElement): string {
  // Scale down canvas to a thumbnail
  const tempCanvas = document.createElement('canvas');
  const aspectRatio = canvas.width / canvas.height;
  tempCanvas.width = 500; // Fixed width for thumbnail
  tempCanvas.height = Math.round(500 / aspectRatio);
  
  const tempCtx = tempCanvas.getContext('2d');
  if (!tempCtx) return '';
  
  // Draw the original canvas onto the thumbnail canvas
  tempCtx.drawImage(canvas, 0, 0, tempCanvas.width, tempCanvas.height);
  
  // Return as base64 data URL
  return tempCanvas.toDataURL('image/png');
}

// Drawing history type for undo/redo functionality
export interface HistoryItem {
  imageData: ImageData;
}

export function createEmptyImageData(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas context not available');
  return ctx.createImageData(canvas.width, canvas.height);
}
