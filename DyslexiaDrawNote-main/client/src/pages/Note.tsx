import { useEffect, useState, useRef } from 'react';
import { useParams, useLocation } from 'wouter';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import DrawingCanvas from '@/components/DrawingCanvas';
import TextRecognition from '@/components/TextRecognition';
import CustomOcrTrainer from '@/components/CustomOcrTrainer';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import {
  ArrowLeft, Save, Share, BrainCircuit,
  PencilLine, LayoutTemplate, Settings,
  Maximize2, Minimize2,
} from 'lucide-react';
import { apiRequest } from '@/lib/queryClient';
import { queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { getCanvasPreview } from '@/lib/utils';
import type { Note as NoteType } from '@shared/schema';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface StrokePoint {
  x: number;
  y: number;
  time: number;
  pen_down: boolean;
  pressure?: number;
  stroke_id?: string;
}

const Note = () => {
  const { id } = useParams<{ id: string }>();
  const [, navigate] = useLocation();
  const { toast } = useToast();

  const [title, setTitle] = useState('Untitled Note');
  const [content, setContent] = useState('');
  const [preview, setPreview] = useState('');
  const [recognizedText, setRecognizedText] = useState('');
  const [lastSavedAt, setLastSavedAt] = useState<Date | null>(null);
  const [splitView, setSplitView] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [activeMode, setActiveMode] = useState<'free' | 'notebook' | 'training'>('free');
  const [autoCorrectShapes, setAutoCorrectShapes] = useState(true);
  const [instantCorrection, setInstantCorrection] = useState(true);
  const [lineSpacing, setLineSpacing] = useState<'single' | 'wide' | 'college'>('single');
  const [backgroundStyle, setBackgroundStyle] = useState<'blank' | 'lined' | 'graph'>('lined');
  const [strokeData, setStrokeData] = useState<StrokePoint[]>([]);

  const { data: noteData, isLoading, error } = useQuery<NoteType>({
    queryKey: id ? [`/api/notes/${id}`] : null,
    enabled: !!id,
  });

  useEffect(() => {
    if (noteData) {
      setTitle(noteData.title);
      setContent(noteData.content);
      setRecognizedText(noteData.recognizedText || '');
    }
  }, [noteData]);

  const handleCanvasReady = (canvas: HTMLCanvasElement) => {
    canvasRef.current = canvas;
  };

  const handleContentChange = (newContent: string) => {
    setContent(newContent);
    if (canvasRef.current) {
      setPreview(getCanvasPreview(canvasRef.current));
    }
  };

  const handleStrokeDataChange = (strokes: StrokePoint[]) => {
    setStrokeData(strokes);
  };

  const handleTextRecognized = (text: string) => {
    setRecognizedText(text);
  };

  const saveMutation = useMutation({
    mutationFn: async () => {
      const notePayload = {
        title,
        content,
        preview,
        recognizedText,
        isFavorite: noteData?.isFavorite || false,
      };

      if (id) {
        await apiRequest('PUT', `/api/notes/${id}`, notePayload);
      } else {
        await apiRequest('POST', '/api/notes', notePayload);
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
    },
  });

  const renderModeSettings = () => {
    switch (activeMode) {
      case 'free':
        return (
          <div className="flex flex-col space-y-4 md:space-y-0 md:flex-row md:items-center md:justify-between p-3 bg-slate-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Switch id="auto-correct-shapes" checked={autoCorrectShapes} onCheckedChange={setAutoCorrectShapes} />
              <Label htmlFor="auto-correct-shapes" className="font-dyslexic text-sm">Auto-correct shapes</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="background-style" className="font-dyslexic text-sm">Background:</Label>
              <Select value={backgroundStyle} onValueChange={(v: 'blank' | 'lined' | 'graph') => setBackgroundStyle(v)}>
                <SelectTrigger id="background-style" className="w-[130px]">
                  <SelectValue placeholder="Background" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="blank">Blank</SelectItem>
                  <SelectItem value="lined">Lined</SelectItem>
                  <SelectItem value="graph">Graph</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        );

      case 'notebook':
        return (
          <div className="flex flex-col space-y-4 md:space-y-0 md:flex-row md:items-center md:justify-between p-3 bg-slate-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Switch id="instant-correction" checked={instantCorrection} onCheckedChange={setInstantCorrection} />
              <Label htmlFor="instant-correction" className="font-dyslexic text-sm">Instant text correction</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="line-spacing" className="font-dyslexic text-sm">Line spacing:</Label>
              <Select value={lineSpacing} onValueChange={(v: 'single' | 'wide' | 'college') => setLineSpacing(v)}>
                <SelectTrigger id="line-spacing" className="w-[130px]">
                  <SelectValue placeholder="Spacing" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="single">Single</SelectItem>
                  <SelectItem value="wide">Wide ruled</SelectItem>
                  <SelectItem value="college">College ruled</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        );

      case 'training':
        return (
          <div className="p-3 bg-slate-50 rounded-lg">
            <p className="text-sm text-muted-foreground font-dyslexic">
              Train the OCR model to better recognise your handwriting.
            </p>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div>
      {/* Header bar */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-4">
        <div className="flex items-center mb-4 md:mb-0">
          <Button variant="ghost" className="mr-3 text-gray-600 hover:text-primary" onClick={() => navigate('/')}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <Input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="text-xl font-bold font-dyslexic bg-transparent border-b border-transparent focus:border-primary focus:ring-0 py-1 px-2"
            placeholder="Untitled Note"
          />
        </div>

        <div className="flex space-x-2 items-center">
          {/* Split view toggle */}
          {activeMode !== 'training' && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSplitView(v => !v)}
              title={splitView ? 'Single panel' : 'Split panel'}
            >
              {splitView ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              <span className="ml-1 hidden sm:inline font-dyslexic text-xs">
                {splitView ? 'Single' : 'Split'}
              </span>
            </Button>
          )}

          <Button
            onClick={() => saveMutation.mutate()}
            className="bg-secondary text-white font-dyslexic"
            disabled={saveMutation.isPending}
            size="sm"
          >
            <Save className="mr-1 h-4 w-4" />
            {saveMutation.isPending ? 'Saving…' : 'Save'}
          </Button>

          <Button variant="outline" size="sm" className="text-gray-700 font-dyslexic">
            <Share className="mr-1 h-4 w-4" />
            Share
          </Button>
        </div>
      </div>

      {isLoading && id && (
        <div className="flex justify-center items-center min-h-[60vh]">
          <p className="font-dyslexic text-lg">Loading note…</p>
        </div>
      )}

      {error && id && (
        <div className="flex justify-center items-center min-h-[60vh]">
          <p className="font-dyslexic text-lg text-red-500">Error loading note. Please try again.</p>
        </div>
      )}

      {(!isLoading || !id) && (
        <Tabs
          defaultValue={activeMode}
          className="w-full"
          onValueChange={(value) => setActiveMode(value as 'free' | 'notebook' | 'training')}
        >
          <div className="flex justify-between items-center mb-2">
            <TabsList>
              <TabsTrigger value="free" className="flex items-center text-xs sm:text-sm">
                <PencilLine className="mr-1 h-4 w-4" /> Free Drawing
              </TabsTrigger>
              <TabsTrigger value="notebook" className="flex items-center text-xs sm:text-sm">
                <LayoutTemplate className="mr-1 h-4 w-4" /> Notebook
              </TabsTrigger>
              <TabsTrigger value="training" className="flex items-center text-xs sm:text-sm">
                <BrainCircuit className="mr-1 h-4 w-4" /> Training
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Mode settings strip */}
          <div className="mb-3">{renderModeSettings()}</div>

          {/* Free Drawing */}
          <TabsContent value="free">
            {splitView ? (
              <ResizablePanelGroup direction="horizontal" className="min-h-[520px] rounded-lg border">
                <ResizablePanel defaultSize={60} minSize={35}>
                  <div className="h-full p-1">
                    <DrawingCanvas
                      initialContent={content}
                      onContentChange={handleContentChange}
                      onStrokeDataChange={handleStrokeDataChange}
                      onCanvasReady={handleCanvasReady}
                      backgroundStyle={backgroundStyle}
                      enableShapeCorrection={autoCorrectShapes}
                      mode="free"
                    />
                  </div>
                </ResizablePanel>
                <ResizableHandle withHandle />
                <ResizablePanel defaultSize={40} minSize={25}>
                  <div className="h-full overflow-y-auto p-3">
                    <TextRecognition
                      canvasElement={canvasRef.current}
                      onTextRecognized={handleTextRecognized}
                    />
                  </div>
                </ResizablePanel>
              </ResizablePanelGroup>
            ) : (
              <>
                <DrawingCanvas
                  initialContent={content}
                  onContentChange={handleContentChange}
                  onStrokeDataChange={handleStrokeDataChange}
                  onCanvasReady={handleCanvasReady}
                  backgroundStyle={backgroundStyle}
                  enableShapeCorrection={autoCorrectShapes}
                  mode="free"
                />
                <div className="mt-4 bg-white rounded-lg shadow p-4">
                  <TextRecognition
                    canvasElement={canvasRef.current}
                    onTextRecognized={handleTextRecognized}
                  />
                </div>
              </>
            )}
          </TabsContent>

          {/* Notebook Mode */}
          <TabsContent value="notebook">
            {splitView ? (
              <ResizablePanelGroup direction="horizontal" className="min-h-[520px] rounded-lg border">
                <ResizablePanel defaultSize={60} minSize={35}>
                  <div className="h-full p-1">
                    <DrawingCanvas
                      initialContent={content}
                      onContentChange={handleContentChange}
                      onStrokeDataChange={handleStrokeDataChange}
                      onCanvasReady={handleCanvasReady}
                      backgroundStyle="lined"
                      lineSpacing={lineSpacing}
                      enableInstantCorrection={instantCorrection}
                      mode="notebook"
                    />
                  </div>
                </ResizablePanel>
                <ResizableHandle withHandle />
                <ResizablePanel defaultSize={40} minSize={25}>
                  <div className="h-full overflow-y-auto p-3">
                    <div className="bg-white rounded-lg shadow p-4 h-full">
                      <h3 className="font-dyslexic font-medium mb-2 text-sm text-gray-700">Corrected Text:</h3>
                      <div className="min-h-[200px] p-3 bg-slate-50 rounded border font-dyslexic text-base leading-loose">
                        {recognizedText || 'Write on the lines to see instant text correction'}
                      </div>
                    </div>
                  </div>
                </ResizablePanel>
              </ResizablePanelGroup>
            ) : (
              <>
                <DrawingCanvas
                  initialContent={content}
                  onContentChange={handleContentChange}
                  onStrokeDataChange={handleStrokeDataChange}
                  onCanvasReady={handleCanvasReady}
                  backgroundStyle="lined"
                  lineSpacing={lineSpacing}
                  enableInstantCorrection={instantCorrection}
                  mode="notebook"
                />
                <div className="mt-4 bg-white rounded-lg shadow p-4">
                  <h3 className="font-medium mb-2 font-dyslexic">Corrected Text:</h3>
                  <div className="min-h-[100px] p-3 bg-slate-50 rounded border font-dyslexic text-lg leading-relaxed">
                    {recognizedText || 'Write on the lines above to see instant text correction'}
                  </div>
                </div>
              </>
            )}
          </TabsContent>

          {/* Training Mode */}
          <TabsContent value="training">
            <div className="bg-white rounded-lg shadow p-4">
              <CustomOcrTrainer canvasElement={canvasRef.current} />
            </div>
          </TabsContent>
        </Tabs>
      )}

      {lastSavedAt && (
        <div className="mt-3 text-right text-xs text-gray-400 font-dyslexic">
          Last saved: {lastSavedAt.toLocaleTimeString()}
        </div>
      )}
    </div>
  );
};

export default Note;
