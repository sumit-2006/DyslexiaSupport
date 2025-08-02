import { useRef, useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import ColorPicker from './ColorPicker';
import PenSizePicker from './PenSizePicker';
import { HistoryItem, createEmptyImageData } from '@/lib/utils';
import { 
  Eraser, 
  Undo2, 
  Redo2, 
  Trash2, 
  Type,
  Shapes,
  Edit3,
  Pen
} from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';

interface DrawingCanvasProps {
  initialContent?: string;
  onContentChange?: (content: string) => void;
  onCanvasReady?: (canvas: HTMLCanvasElement) => void;
}

interface Point {
  x: number;
  y: number;
  pressure?: number; // For pen pressure sensitivity
}

const DrawingCanvas = ({ 
  initialContent,
  onContentChange,
  onCanvasReady
}: DrawingCanvasProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentTool, setCurrentTool] = useState<'pen' | 'eraser' | 'stylus'>('pen');
  const [penColor, setPenColor] = useState('#000000');
  const [penSize, setPenSize] = useState(2);
  const [lastPosition, setLastPosition] = useState<Point>({ x: 0, y: 0 });
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [isPenTabletDetected, setIsPenTabletDetected] = useState(false);
  
  // Initialize canvas and set up event listeners
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = canvasContainerRef.current;
    
    if (!canvas || !container) return;
    
    // Set canvas dimensions
    const resizeCanvas = () => {
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      // If we have history, restore the last state
      if (historyIndex >= 0 && history[historyIndex]) {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.putImageData(history[historyIndex].imageData, 0, 0);
        }
      }
    };
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // If there's initial content, load it
    if (initialContent) {
      const img = new Image();
      img.onload = () => {
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0);
          saveHistoryState();
        }
      };
      img.src = initialContent;
    } else {
      // Save initial blank state
      saveHistoryState();
    }
    
    // Enable pointer events for tablet support (optional, as we now have mouse support too)
    // Check for stylus capability - define this first so we can reference it in cleanup
    const detectPenTablet = (e: PointerEvent) => {
      if (e.pointerType === 'pen') {
        setIsPenTabletDetected(true);
        setCurrentTool('stylus');
        
        // Once detected, remove this listener
        window.removeEventListener('pointerdown', detectPenTablet);
      }
    };
    
    try {
      // Add pointer event listeners
      canvas.addEventListener('pointerdown', handlePointerDown);
      canvas.addEventListener('pointermove', handlePointerMove);
      canvas.addEventListener('pointerup', handlePointerUp);
      canvas.addEventListener('pointerout', handlePointerUp);
      canvas.addEventListener('pointercancel', handlePointerUp);
      
      // Enable touch action
      canvas.style.touchAction = 'none';
      
      // Listen for pen tablets
      window.addEventListener('pointerdown', detectPenTablet);
    } catch (err) {
      console.log('Pointer events not fully supported, using mouse events as fallback');
    }
    
    // Notify parent component that canvas is ready
    if (onCanvasReady) {
      onCanvasReady(canvas);
    }
    
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      
      // Clean up all event listeners
      try {
        canvas.removeEventListener('pointerdown', handlePointerDown);
        canvas.removeEventListener('pointermove', handlePointerMove);
        canvas.removeEventListener('pointerup', handlePointerUp);
        canvas.removeEventListener('pointerout', handlePointerUp);
        canvas.removeEventListener('pointercancel', handlePointerUp);
        window.removeEventListener('pointerdown', detectPenTablet);
      } catch (err) {
        console.log('Error cleaning up pointer events:', err);
      }
    };
  }, []);

  // Save current canvas state to history
  const saveHistoryState = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // If we're not at the end of the history, truncate it
    if (historyIndex < history.length - 1) {
      setHistory(prev => prev.slice(0, historyIndex + 1));
    }
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    setHistory(prev => [...prev, { imageData }]);
    setHistoryIndex(prev => prev + 1);
    
    // Notify parent of content change
    if (onContentChange) {
      onContentChange(canvas.toDataURL());
    }
  };
  
  // Undo function
  const handleUndo = () => {
    if (historyIndex <= 0) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    setHistoryIndex(prev => prev - 1);
    ctx.putImageData(history[historyIndex - 1].imageData, 0, 0);
    
    // Notify parent of content change
    if (onContentChange) {
      onContentChange(canvas.toDataURL());
    }
  };
  
  // Redo function
  const handleRedo = () => {
    if (historyIndex >= history.length - 1) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    setHistoryIndex(prev => prev + 1);
    ctx.putImageData(history[historyIndex + 1].imageData, 0, 0);
    
    // Notify parent of content change
    if (onContentChange) {
      onContentChange(canvas.toDataURL());
    }
  };
  
  // Clear canvas
  const handleClear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    saveHistoryState();
  };
  
  // Pointer events handlers (for pen tablet support)
  const handlePointerDown = (e: PointerEvent) => {
    setIsDrawing(true);
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Capture pointer to ensure all events are directed to this element
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    
    const rect = canvas.getBoundingClientRect();
    const point: Point = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      pressure: e.pressure || 1 // Default to 1 if pressure is not supported
    };
    
    // If it's a stylus, automatically switch to stylus tool
    if (e.pointerType === 'pen' && currentTool !== 'eraser') {
      setCurrentTool('stylus');
    }
    
    setLastPosition(point);
    
    // Start a new path for this stroke
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.beginPath();
      drawPoint(ctx, point);
    }
  };
  
  const handlePointerMove = (e: PointerEvent) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const rect = canvas.getBoundingClientRect();
    const point: Point = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      pressure: e.pressure || 1
    };
    
    // Draw a line from last position to current position
    drawLine(ctx, lastPosition, point);
    
    setLastPosition(point);
  };
  
  const handlePointerUp = (e: PointerEvent) => {
    if (isDrawing) {
      // Release pointer capture
      if ((e.target as HTMLElement).hasPointerCapture?.(e.pointerId)) {
        (e.target as HTMLElement).releasePointerCapture(e.pointerId);
      }
      
      setIsDrawing(false);
      saveHistoryState();
    }
  };
  
  // Mouse event handlers (fallback for non-pointer devices)
  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Allow both pointer events and mouse events to work
    // This ensures backward compatibility with all devices
    setIsDrawing(true);
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setLastPosition({ x, y });
    
    // Start a new path
    const ctx = canvas.getContext('2d');
    if (ctx) {
      configureContext(ctx);
      ctx.beginPath();
      ctx.arc(x, y, penSize / 2, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  
  const draw = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Draw line from last position to current position
    drawLine(ctx, lastPosition, { x, y });
    
    setLastPosition({ x, y });
  };
  
  const stopDrawing = () => {
    if (isDrawing) {
      setIsDrawing(false);
      saveHistoryState();
    }
  };
  
  // Drawing helper functions
  const drawPoint = (ctx: CanvasRenderingContext2D, point: Point) => {
    configureContext(ctx);
    
    // Draw a single point (useful for dots and small strokes)
    ctx.beginPath();
    ctx.arc(point.x, point.y, getAdjustedPenSize(point.pressure) / 2, 0, Math.PI * 2);
    ctx.fill();
  };
  
  const drawLine = (ctx: CanvasRenderingContext2D, from: Point, to: Point) => {
    configureContext(ctx);
    
    // Draw a line from 'from' to 'to' points
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    
    // For stylus, vary line width based on pressure
    if (currentTool === 'stylus' && to.pressure !== undefined) {
      ctx.lineWidth = getAdjustedPenSize(to.pressure);
    }
    
    ctx.stroke();
  };
  
  const configureContext = (ctx: CanvasRenderingContext2D) => {
    // Set common drawing properties
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    switch(currentTool) {
      case 'pen':
        ctx.strokeStyle = penColor;
        ctx.fillStyle = penColor;
        ctx.lineWidth = penSize;
        ctx.globalCompositeOperation = 'source-over';
        break;
        
      case 'stylus':
        ctx.strokeStyle = penColor;
        ctx.fillStyle = penColor;
        ctx.lineWidth = penSize;
        ctx.globalCompositeOperation = 'source-over';
        break;
        
      case 'eraser':
        ctx.strokeStyle = '#ffffff';
        ctx.fillStyle = '#ffffff';
        ctx.lineWidth = penSize * 2; // Eraser is typically larger
        ctx.globalCompositeOperation = 'destination-out';
        break;
    }
  };
  
  // Adjust pen size based on pressure for stylus
  const getAdjustedPenSize = (pressure: number = 1): number => {
    if (currentTool === 'stylus') {
      // Scale pen size based on pressure (0.5 to 2x the base size)
      const minFactor = 0.5;
      const maxFactor = 2.0;
      const factor = minFactor + pressure * (maxFactor - minFactor);
      return penSize * factor;
    }
    return penSize;
  };
  
  // Handle tool changes
  const handleToolChange = (tool: 'pen' | 'eraser' | 'stylus') => {
    setCurrentTool(tool);
  };
  
  // Handle pen color change
  const handleColorChange = (color: string) => {
    setPenColor(color);
    // Keep the current tool if it's stylus or pen
    if (currentTool === 'eraser') {
      setCurrentTool(isPenTabletDetected ? 'stylus' : 'pen');
    }
  };
  
  // Handle pen size change
  const handleSizeChange = (size: number) => {
    setPenSize(size);
  };

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg shadow-md p-3 mb-4 flex flex-wrap items-center justify-between">
        {/* Left Tools Group */}
        <div className="flex items-center space-x-2 mb-2 md:mb-0">
          {/* Pen Tools */}
          <div className="border-r pr-2 mr-2">
            <PenSizePicker onSizeChange={handleSizeChange} defaultSize={penSize} />
          </div>
          
          {/* Color Picker */}
          <ColorPicker onColorChange={handleColorChange} defaultColor={penColor} />
        </div>
        
        {/* Input Method Indicator */}
        {isPenTabletDetected && (
          <Badge variant="outline" className="bg-green-50 text-green-700 mr-2">
            Pen Tablet Detected
          </Badge>
        )}
        
        {/* Right Tools Group */}
        <div className="flex items-center space-x-3">
          {/* Pen/Stylus Button */}
          <Button
            variant={(currentTool === 'pen' || currentTool === 'stylus') ? 'secondary' : 'outline'}
            size="icon"
            onClick={() => handleToolChange(isPenTabletDetected ? 'stylus' : 'pen')}
            title={isPenTabletDetected ? "Stylus" : "Pen"}
          >
            {isPenTabletDetected ? (
              <Edit3 className="h-5 w-5" />
            ) : (
              <Pen className="h-5 w-5" />
            )}
          </Button>
          
          {/* Eraser Button */}
          <Button
            variant={currentTool === 'eraser' ? 'secondary' : 'outline'}
            size="icon"
            onClick={() => handleToolChange('eraser')}
            title="Eraser"
          >
            <Eraser className="h-5 w-5" />
          </Button>
          
          <Button
            variant="outline"
            size="icon"
            title="Text Tool"
          >
            <Type className="h-5 w-5" />
          </Button>
          
          <Button
            variant="outline"
            size="icon"
            title="Shapes"
          >
            <Shapes className="h-5 w-5" />
          </Button>
          
          <Separator orientation="vertical" className="h-8" />
          
          <Button
            variant="outline"
            size="icon"
            onClick={handleUndo}
            disabled={historyIndex <= 0}
            title="Undo"
          >
            <Undo2 className="h-5 w-5" />
          </Button>
          
          <Button
            variant="outline"
            size="icon"
            onClick={handleRedo}
            disabled={historyIndex >= history.length - 1}
            title="Redo"
          >
            <Redo2 className="h-5 w-5" />
          </Button>
          
          <Separator orientation="vertical" className="h-8" />
          
          <Button
            variant="outline"
            size="icon"
            onClick={handleClear}
            title="Clear Canvas"
          >
            <Trash2 className="h-5 w-5" />
          </Button>
        </div>
      </div>
      
      <div 
        ref={canvasContainerRef}
        className="bg-white rounded-lg shadow-lg p-1 overflow-hidden canvas-container"
        style={{ height: '70vh' }}
      >
        <canvas
          ref={canvasRef}
          className="w-full border border-gray-200 rounded-lg bg-white"
          style={{ touchAction: 'none' }} // Disable browser handling of touch events
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
        />
      </div>
    </div>
  );
};

export default DrawingCanvas;
