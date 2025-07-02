import { useEffect, useState, useRef } from 'react';
import { useParams, useLocation } from 'wouter';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import DrawingCanvas from '@/components/DrawingCanvas';
import TextRecognition from '@/components/TextRecognition';
import { ArrowLeft, Save, Share } from 'lucide-react';
import { apiRequest } from '@/lib/queryClient';
import { queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { getCanvasPreview } from '@/lib/utils';
import type { Note as NoteType } from '@shared/schema';

const Note = () => {
  const { id } = useParams<{ id: string }>();
  const [, navigate] = useLocation();
  const { toast } = useToast();
  
  const [title, setTitle] = useState('Untitled Note');
  const [content, setContent] = useState('');
  const [preview, setPreview] = useState('');
  const [recognizedText, setRecognizedText] = useState('');
  const [lastSavedAt, setLastSavedAt] = useState<Date | null>(null);
  
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Fetch note data if editing an existing note
  const {
    data: noteData,
    isLoading,
    error
  } = useQuery<NoteType>({
    queryKey: id ? [`/api/notes/${id}`] : null,
    enabled: !!id,
  });

  // Effect to set initial data from loaded note
  useEffect(() => {
    if (noteData) {
      setTitle(noteData.title);
      setContent(noteData.content);
      setRecognizedText(noteData.recognizedText || '');
    }
  }, [noteData]);

  // Handle canvas ready event
  const handleCanvasReady = (canvas: HTMLCanvasElement) => {
    canvasRef.current = canvas;
  };

  // Handle canvas content change
  const handleContentChange = (newContent: string) => {
    setContent(newContent);
    
    // Generate preview when content changes
    if (canvasRef.current) {
      setPreview(getCanvasPreview(canvasRef.current));
    }
  };

  // Handle recognized text from TextRecognition component
  const handleTextRecognized = (text: string) => {
    setRecognizedText(text);
  };

  // Save mutation
  const saveMutation = useMutation({
    mutationFn: async () => {
      const noteData = {
        title,
        content,
        preview,
        recognizedText,
        isFavorite: noteData?.isFavorite || false
      };

      if (id) {
        // Update existing note
        await apiRequest('PUT', `/api/notes/${id}`, noteData);
      } else {
        // Create new note
        await apiRequest('POST', '/api/notes', noteData);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/notes'] });
      if (id) {
        queryClient.invalidateQueries({ queryKey: [`/api/notes/${id}`] });
      }
      
      setLastSavedAt(new Date());
      
      toast({
        title: id ? 'Note updated' : 'Note created',
        description: `"${title}" has been ${id ? 'updated' : 'saved'} successfully.`,
      });
      
      // If this is a new note, navigate to home after saving
      if (!id) {
        navigate('/');
      }
    },
    onError: () => {
      toast({
        title: 'Error saving note',
        description: 'Please try again later.',
        variant: 'destructive',
      });
    }
  });

  // Function to handle save button click
  const handleSave = () => {
    saveMutation.mutate();
  };

  return (
    <div>
      {/* Note Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <div className="flex items-center mb-4 md:mb-0">
          <Button 
            variant="ghost"
            className="mr-4 text-gray-600 hover:text-primary"
            onClick={() => navigate('/')}
          >
            <ArrowLeft className="h-6 w-6" />
          </Button>
          
          <Input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="text-2xl font-bold font-dyslexic bg-transparent border-b border-transparent focus:border-primary focus:ring-0 py-1 px-2 w-auto"
            placeholder="Untitled Note"
          />
        </div>
        
        <div className="flex space-x-3">
          <Button
            onClick={handleSave}
            className="bg-secondary text-white font-dyslexic flex items-center"
            disabled={saveMutation.isPending}
          >
            <Save className="mr-2 h-5 w-5" />
            {saveMutation.isPending ? 'Saving...' : 'Save'}
          </Button>
          
          <Button
            variant="outline"
            className="text-gray-700 font-dyslexic flex items-center"
          >
            <Share className="mr-2 h-5 w-5" />
            Share
          </Button>
        </div>
      </div>

      {/* Loading state */}
      {isLoading && id && (
        <div className="flex justify-center items-center min-h-[70vh]">
          <p className="font-dyslexic text-lg">Loading note...</p>
        </div>
      )}

      {/* Error state */}
      {error && id && (
        <div className="flex justify-center items-center min-h-[70vh]">
          <p className="font-dyslexic text-lg text-red-500">
            Error loading note. Please try again later.
          </p>
        </div>
      )}

      {/* Note content */}
      {(!isLoading || !id) && (
        <div className="relative">
          {/* Drawing Canvas */}
          <DrawingCanvas
            initialContent={content}
            onContentChange={handleContentChange}
            onCanvasReady={handleCanvasReady}
          />
          
          {/* Text Recognition Panel */}
          <TextRecognition
            canvasElement={canvasRef.current}
            onTextRecognized={handleTextRecognized}
          />
        </div>
      )}
      
      {/* Last saved information */}
      {lastSavedAt && (
        <div className="mt-4 text-right text-sm text-gray-500 font-dyslexic">
          Last saved: {lastSavedAt.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default Note;
