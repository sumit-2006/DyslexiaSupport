import { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { recognizeText, aiCorrectText } from '@/lib/tesseract';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Loader2, FileText, Edit, Volume2, VolumeX, Sparkles, RefreshCw } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { useToast } from '@/hooks/use-toast';

interface TextRecognitionProps {
  canvasElement: HTMLCanvasElement | null;
  onTextRecognized?: (text: string) => void;
}

function getWhiteBackgroundImage(canvas: HTMLCanvasElement): string {
  const tempCanvas = document.createElement('canvas');
  const ctx = tempCanvas.getContext('2d')!;
  tempCanvas.width = canvas.width;
  tempCanvas.height = canvas.height;
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
  ctx.drawImage(canvas, 0, 0);
  return tempCanvas.toDataURL('image/png');
}

const TextRecognition = ({ canvasElement, onTextRecognized }: TextRecognitionProps) => {
  const { toast } = useToast();
  const [recognizedText, setRecognizedText] = useState('');
  const [formattedText, setFormattedText] = useState('');
  const [correctedText, setCorrectedText] = useState('');
  const [suggestions, setSuggestions] = useState<{ original: string; correction: string }[]>([]);
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [isAiCorrecting, setIsAiCorrecting] = useState(false);
  const [recognitionProgress, setRecognitionProgress] = useState(0);
  const [activeTab, setActiveTab] = useState('handwritten');
  const [autoRecognize, setAutoRecognize] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const autoRecognizeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounced auto-recognition
  useEffect(() => {
    if (!autoRecognize || !canvasElement) return;

    if (autoRecognizeTimer.current) {
      clearTimeout(autoRecognizeTimer.current);
    }

    autoRecognizeTimer.current = setTimeout(() => {
      handleRecognizeText();
    }, 2500);

    return () => {
      if (autoRecognizeTimer.current) clearTimeout(autoRecognizeTimer.current);
    };
  }, [autoRecognize, canvasElement]);

  const handleRecognizeText = useCallback(async () => {
    if (!canvasElement) return;

    setIsRecognizing(true);
    setRecognitionProgress(10);

    try {
      const imageData = getWhiteBackgroundImage(canvasElement);
      setRecognitionProgress(30);

      const result = await recognizeText(imageData);

      setRecognizedText(result.text);
      setFormattedText(result.formattedText);
      setSuggestions(result.suggestions);
      setCorrectedText(''); // reset AI correction when new OCR runs

      if (result.formattedText && result.text.length > 5) {
        setActiveTab('computerfont');
      }

      if (onTextRecognized) {
        onTextRecognized(result.formattedText || result.text);
      }

      setRecognitionProgress(100);
    } catch (error) {
      console.error('Error recognizing text:', error);
      toast({ title: 'Recognition failed', description: 'Please try again.', variant: 'destructive' });
    } finally {
      setIsRecognizing(false);
    }
  }, [canvasElement, onTextRecognized, toast]);

  const handleAiCorrect = async () => {
    const textToCorrect = formattedText || recognizedText;
    if (!textToCorrect) return;

    setIsAiCorrecting(true);
    try {
      const result = await aiCorrectText(textToCorrect);
      setCorrectedText(result.correctedText);
      setSuggestions(result.suggestions);
      setActiveTab('ai');

      if (onTextRecognized) {
        onTextRecognized(result.correctedText);
      }
    } catch (error) {
      toast({ title: 'AI correction failed', description: 'Please try again.', variant: 'destructive' });
    } finally {
      setIsAiCorrecting(false);
    }
  };

  const handleSpeak = (text: string) => {
    if (!('speechSynthesis' in window)) {
      toast({ title: 'TTS not supported', description: 'Your browser does not support text-to-speech.' });
      return;
    }

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.85;
    utterance.pitch = 1;
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
    setIsSpeaking(true);
  };

  const applySuggestion = (original: string, correction: string) => {
    const updatedText = recognizedText.replace(new RegExp(original, 'gi'), correction);
    const updatedFormatted = formattedText.replace(new RegExp(original, 'gi'), correction);

    setRecognizedText(updatedText);
    setFormattedText(updatedFormatted);
    setSuggestions(prev => prev.filter(s => s.original !== original));

    if (onTextRecognized) {
      onTextRecognized(activeTab === 'computerfont' ? updatedFormatted : updatedText);
    }
  };

  const activeText =
    activeTab === 'ai' ? correctedText :
    activeTab === 'computerfont' ? formattedText :
    recognizedText;

  const hasText = !!(recognizedText || formattedText);

  return (
    <Card className="mt-4">
      <CardHeader className="pb-3">
        <div className="flex flex-wrap justify-between items-center gap-2">
          <CardTitle className="font-dyslexic text-lg">Text Recognition</CardTitle>

          <div className="flex items-center gap-2 flex-wrap">
            {/* Auto-recognize toggle */}
            <div className="flex items-center gap-1.5">
              <Switch
                id="auto-recognize"
                checked={autoRecognize}
                onCheckedChange={setAutoRecognize}
                className="scale-90"
              />
              <Label htmlFor="auto-recognize" className="text-xs font-dyslexic cursor-pointer">
                Auto
              </Label>
            </div>

            {/* TTS button */}
            {hasText && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleSpeak(activeText)}
                title={isSpeaking ? 'Stop speaking' : 'Read aloud'}
              >
                {isSpeaking ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
              </Button>
            )}

            {/* AI Correct button */}
            {hasText && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleAiCorrect}
                disabled={isAiCorrecting}
                title="AI-powered dyslexia correction"
              >
                {isAiCorrecting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                <span className="ml-1 hidden sm:inline">AI Fix</span>
              </Button>
            )}

            {/* Recognize button */}
            <Button
              onClick={handleRecognizeText}
              disabled={!canvasElement || isRecognizing}
              size="sm"
            >
              {isRecognizing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {recognitionProgress > 0 ? `${recognitionProgress}%` : 'Processing...'}
                </>
              ) : (
                <>
                  <RefreshCw className="mr-1 h-4 w-4" />
                  Recognize
                </>
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {hasText ? (
          <Tabs defaultValue={activeTab} value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className={`grid w-full ${correctedText ? 'grid-cols-3' : 'grid-cols-2'}`}>
              <TabsTrigger value="handwritten" className="flex items-center text-xs sm:text-sm">
                <Edit className="mr-1 h-3 w-3 sm:h-4 sm:w-4" />
                Raw
              </TabsTrigger>
              <TabsTrigger value="computerfont" className="flex items-center text-xs sm:text-sm">
                <FileText className="mr-1 h-3 w-3 sm:h-4 sm:w-4" />
                Clean
              </TabsTrigger>
              {correctedText && (
                <TabsTrigger value="ai" className="flex items-center text-xs sm:text-sm">
                  <Sparkles className="mr-1 h-3 w-3 sm:h-4 sm:w-4" />
                  AI Fixed
                </TabsTrigger>
              )}
            </TabsList>

            <TabsContent value="handwritten" className="mt-2">
              <div className="font-dyslexic text-gray-800 p-3 bg-gray-50 rounded-lg min-h-24 whitespace-pre-wrap leading-loose tracking-wide">
                {recognizedText}
              </div>
            </TabsContent>

            <TabsContent value="computerfont" className="mt-2">
              <div className="font-sans text-gray-800 p-3 bg-gray-50 rounded-lg min-h-24 whitespace-pre-wrap leading-relaxed">
                {formattedText || 'No clean text available'}
              </div>
            </TabsContent>

            {correctedText && (
              <TabsContent value="ai" className="mt-2">
                <div className="font-sans text-gray-800 p-3 bg-purple-50 rounded-lg min-h-24 whitespace-pre-wrap leading-relaxed border border-purple-200">
                  {correctedText}
                </div>
              </TabsContent>
            )}
          </Tabs>
        ) : (
          <div className="mb-3 font-dyslexic text-gray-500 p-3 bg-gray-50 rounded-lg min-h-24 flex items-center justify-center text-center">
            Draw something and click "Recognize" to see the result
            {autoRecognize && ' (or wait for auto-recognition)'}
          </div>
        )}

        {Array.isArray(suggestions) && suggestions.length > 0 && (
          <>
            <Separator className="my-3" />
            <h4 className="font-dyslexic font-semibold text-sm text-gray-600 mb-2">
              Suggested Corrections:
            </h4>
            <div className="flex flex-wrap gap-2">
              {suggestions.map(({ original, correction }, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  className="font-dyslexic text-xs border-purple-300 text-purple-700 hover:bg-purple-50"
                  onClick={() => applySuggestion(original, correction)}
                >
                  "{original}" → "{correction}"
                </Button>
              ))}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default TextRecognition;

