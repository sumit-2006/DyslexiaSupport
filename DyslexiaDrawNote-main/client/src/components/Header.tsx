import { useState } from 'react';
import { Link, useLocation } from "wouter";
import { Button } from "@/components/ui/button";
import { PlusCircle, ArrowLeft, Brain, Settings2, ZoomIn, ZoomOut, AlignJustify } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";

// Simple global accessibility state stored in localStorage
function getA11yValue(key: string, defaultVal: number): number {
  try {
    const val = localStorage.getItem(key);
    return val !== null ? parseFloat(val) : defaultVal;
  } catch {
    return defaultVal;
  }
}

function setA11yValue(key: string, value: number) {
  try {
    localStorage.setItem(key, String(value));
  } catch {}
}

function applyA11y(fontSize: number, letterSpacing: number, lineHeight: number) {
  document.documentElement.style.setProperty('--a11y-font-size', `${fontSize}px`);
  document.documentElement.style.setProperty('--a11y-letter-spacing', `${letterSpacing}px`);
  document.documentElement.style.setProperty('--a11y-line-height', `${lineHeight}`);
}

const Header = () => {
  const [location, navigate] = useLocation();
  const isNoteView = location.startsWith('/note');
  const isTrainingView = location.startsWith('/training');
  const isAssessmentView = location.startsWith('/assessment');

  const [fontSize, setFontSize] = useState(() => getA11yValue('a11y-font-size', 16));
  const [letterSpacing, setLetterSpacing] = useState(() => getA11yValue('a11y-letter-spacing', 0.5));
  const [lineHeight, setLineHeight] = useState(() => getA11yValue('a11y-line-height', 1.6));

  const handleFontSize = (vals: number[]) => {
    const v = vals[0];
    setFontSize(v);
    setA11yValue('a11y-font-size', v);
    applyA11y(v, letterSpacing, lineHeight);
  };

  const handleLetterSpacing = (vals: number[]) => {
    const v = vals[0];
    setLetterSpacing(v);
    setA11yValue('a11y-letter-spacing', v);
    applyA11y(fontSize, v, lineHeight);
  };

  const handleLineHeight = (vals: number[]) => {
    const v = vals[0];
    setLineHeight(v);
    setA11yValue('a11y-line-height', v);
    applyA11y(fontSize, letterSpacing, v);
  };

  const isBack = isNoteView || isTrainingView || isAssessmentView;

  return (
    <header className="bg-white shadow-md py-4 px-6 mb-6">
      <div className="container mx-auto flex justify-between items-center">
        {/* Logo */}
        <div className="flex items-center">
          <Link href="/">
            <h1 className="text-3xl font-bold text-primary font-dyslexic cursor-pointer">
              DyslexiNote
            </h1>
          </Link>
        </div>

        {/* Desktop nav */}
        <nav className="hidden md:flex space-x-6 items-center mr-4">
          <div className={`text-base ${location === '/' ? 'font-bold text-primary' : 'text-gray-600 hover:text-primary'}`}>
            <Link href="/">Home</Link>
          </div>
          <div className={`text-base ${location === '/training' ? 'font-bold text-primary' : 'text-gray-600 hover:text-primary'}`}>
            <Link href="/training">OCR Training</Link>
          </div>
          <div className={`text-base ${location === '/assessment' ? 'font-bold text-primary' : 'text-gray-600 hover:text-primary'}`}>
            <Link href="/assessment">Assessment</Link>
          </div>
        </nav>

        {/* Actions */}
        <div className="flex items-center space-x-2">
          {/* Accessibility settings */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="ghost" size="icon" title="Accessibility settings">
                <Settings2 className="h-5 w-5 text-gray-500" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-72 p-4 space-y-5">
              <p className="font-dyslexic font-semibold text-sm text-gray-700 flex items-center gap-2">
                <Settings2 className="h-4 w-4" /> Accessibility
              </p>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="font-dyslexic text-xs flex items-center gap-1">
                    <ZoomIn className="h-3 w-3" /> Font Size
                  </Label>
                  <span className="text-xs text-gray-500">{fontSize}px</span>
                </div>
                <Slider
                  min={13}
                  max={24}
                  step={1}
                  value={[fontSize]}
                  onValueChange={handleFontSize}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="font-dyslexic text-xs flex items-center gap-1">
                    <AlignJustify className="h-3 w-3" /> Letter Spacing
                  </Label>
                  <span className="text-xs text-gray-500">{letterSpacing}px</span>
                </div>
                <Slider
                  min={0}
                  max={4}
                  step={0.25}
                  value={[letterSpacing]}
                  onValueChange={handleLetterSpacing}
                />
              </div>

              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <Label className="font-dyslexic text-xs">Line Height</Label>
                  <span className="text-xs text-gray-500">{lineHeight}×</span>
                </div>
                <Slider
                  min={1.2}
                  max={2.5}
                  step={0.1}
                  value={[lineHeight]}
                  onValueChange={handleLineHeight}
                />
              </div>

              <Button
                variant="outline"
                size="sm"
                className="w-full font-dyslexic text-xs"
                onClick={() => {
                  handleFontSize([16]);
                  handleLetterSpacing([0.5]);
                  handleLineHeight([1.6]);
                }}
              >
                Reset to defaults
              </Button>
            </PopoverContent>
          </Popover>

          {isBack ? (
            <Button
              onClick={() => navigate('/')}
              className="font-dyslexic text-sm font-semibold"
            >
              <ArrowLeft className="mr-2 h-5 w-5" /> Back
            </Button>
          ) : (
            <Button
              onClick={() => navigate('/note')}
              className="font-dyslexic text-sm font-semibold"
            >
              <PlusCircle className="mr-2 h-5 w-5" /> New Note
            </Button>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;

