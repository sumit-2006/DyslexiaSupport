import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { CheckCircle2, XCircle, RotateCcw, Brain, ChevronRight, Timer } from 'lucide-react';

// ── Types ─────────────────────────────────────────────────────────────────────

interface ReversalQuestion {
  display: string;       // what the user sees
  options: string[];     // answer choices
  correct: string;       // correct answer
}

interface SpeedRound {
  passage: string;
  wordCount: number;
}

type Phase = 'intro' | 'reversal' | 'speed' | 'results';

// ── Data ──────────────────────────────────────────────────────────────────────

const REVERSAL_QUESTIONS: ReversalQuestion[] = [
  { display: 'b', options: ['b', 'd', 'p', 'q'], correct: 'b' },
  { display: 'd', options: ['b', 'd', 'p', 'q'], correct: 'd' },
  { display: 'p', options: ['b', 'd', 'p', 'q'], correct: 'p' },
  { display: 'q', options: ['b', 'd', 'p', 'q'], correct: 'q' },
  { display: 'n', options: ['n', 'u', 'm', 'h'], correct: 'n' },
  { display: 'u', options: ['n', 'u', 'm', 'h'], correct: 'u' },
  { display: 'was', options: ['was', 'saw', 'raw', 'war'], correct: 'was' },
  { display: 'saw', options: ['was', 'saw', 'raw', 'war'], correct: 'saw' },
  { display: 'on',  options: ['on', 'no', 'now', 'own'],  correct: 'on'  },
  { display: 'no',  options: ['on', 'no', 'now', 'own'],  correct: 'no'  },
  { display: '6', options: ['6', '9', '8', '0'], correct: '6' },
  { display: '9', options: ['6', '9', '8', '0'], correct: '9' },
];

const SPEED_ROUNDS: SpeedRound[] = [
  {
    passage:
      'The quick brown fox jumps over the lazy dog. ' +
      'She saw a big red ball on the ground near the tree.',
    wordCount: 20,
  },
  {
    passage:
      'Reading can be fun when you take your time. ' +
      'Every word tells a small story of its own.',
    wordCount: 18,
  },
  {
    passage:
      'The dog sat on the mat. A big cat ran past the hat. ' +
      'Sam and Pam play in the sun all day long.',
    wordCount: 22,
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function shuffle<T>(arr: T[]): T[] {
  return [...arr].sort(() => Math.random() - 0.5);
}

function wpmRating(wpm: number): { label: string; color: string } {
  if (wpm >= 200) return { label: 'Excellent', color: 'text-green-600' };
  if (wpm >= 130) return { label: 'Good', color: 'text-blue-600' };
  if (wpm >= 80)  return { label: 'Average', color: 'text-yellow-600' };
  return { label: 'Developing', color: 'text-orange-600' };
}

// ── Sub-components ────────────────────────────────────────────────────────────

const ReversalTest = ({
  questions,
  onComplete,
}: {
  questions: ReversalQuestion[];
  onComplete: (score: number, errors: string[]) => void;
}) => {
  const [index, setIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [errors, setErrors] = useState<string[]>([]);
  const [feedback, setFeedback] = useState<'correct' | 'wrong' | null>(null);
  const [shuffledOptions, setShuffledOptions] = useState(() =>
    shuffle(questions[0].options),
  );

  const question = questions[index];

  const handleAnswer = (option: string) => {
    if (feedback) return;
    const correct = option === question.correct;

    if (correct) {
      setScore(s => s + 1);
      setFeedback('correct');
    } else {
      setErrors(e => [...e, `${question.display} → chose "${option}"`]);
      setFeedback('wrong');
    }

    setTimeout(() => {
      const next = index + 1;
      if (next >= questions.length) {
        onComplete(correct ? score + 1 : score, errors);
      } else {
        setIndex(next);
        setShuffledOptions(shuffle(questions[next].options));
        setFeedback(null);
      }
    }, 700);
  };

  return (
    <div className="flex flex-col items-center gap-8">
      <div className="w-full max-w-md">
        <div className="flex justify-between text-sm text-gray-500 font-dyslexic mb-2">
          <span>Question {index + 1} / {questions.length}</span>
          <span>Score: {score}</span>
        </div>
        <Progress value={((index + 1) / questions.length) * 100} className="h-2" />
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={index}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 1.1 }}
          transition={{ duration: 0.25 }}
          className={`w-40 h-40 flex items-center justify-center rounded-2xl text-7xl font-bold shadow-lg select-none ${
            feedback === 'correct'
              ? 'bg-green-100 text-green-700'
              : feedback === 'wrong'
              ? 'bg-red-100 text-red-700'
              : 'bg-white text-gray-800 border-2 border-gray-200'
          }`}
        >
          {question.display}
        </motion.div>
      </AnimatePresence>

      <p className="font-dyslexic text-gray-600 text-sm">
        Which letter / word is shown above?
      </p>

      <div className="grid grid-cols-2 gap-3 w-full max-w-xs">
        {shuffledOptions.map(option => (
          <Button
            key={option}
            variant="outline"
            size="lg"
            className="text-2xl font-bold font-dyslexic h-16"
            onClick={() => handleAnswer(option)}
            disabled={!!feedback}
          >
            {option}
          </Button>
        ))}
      </div>
    </div>
  );
};

const SpeedTest = ({
  rounds,
  onComplete,
}: {
  rounds: SpeedRound[];
  onComplete: (wpm: number) => void;
}) => {
  const [roundIndex, setRoundIndex] = useState(0);
  const [phase, setPhase] = useState<'ready' | 'reading' | 'done'>('ready');
  const [startTime, setStartTime] = useState<number>(0);
  const [elapsed, setElapsed] = useState<number>(0);
  const [allWpms, setAllWpms] = useState<number[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const round = rounds[roundIndex];

  const startReading = () => {
    setPhase('reading');
    const now = Date.now();
    setStartTime(now);
    timerRef.current = setInterval(() => {
      setElapsed(Date.now() - now);
    }, 100);
  };

  const doneReading = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    const minutes = elapsed / 1000 / 60;
    const wpm = Math.round(round.wordCount / minutes);
    const newWpms = [...allWpms, wpm];
    setAllWpms(newWpms);
    setPhase('done');

    setTimeout(() => {
      const next = roundIndex + 1;
      if (next >= rounds.length) {
        const avg = Math.round(newWpms.reduce((a, b) => a + b, 0) / newWpms.length);
        onComplete(avg);
      } else {
        setRoundIndex(next);
        setPhase('ready');
        setElapsed(0);
      }
    }, 1200);
  };

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  const seconds = (elapsed / 1000).toFixed(1);
  const currentWpm = elapsed > 0 ? Math.round(round.wordCount / (elapsed / 1000 / 60)) : 0;

  return (
    <div className="flex flex-col items-center gap-6 max-w-lg mx-auto">
      <div className="w-full">
        <div className="flex justify-between text-sm text-gray-500 font-dyslexic mb-2">
          <span>Passage {roundIndex + 1} / {rounds.length}</span>
          {phase === 'reading' && (
            <span className="flex items-center gap-1">
              <Timer className="h-4 w-4" /> {seconds}s
            </span>
          )}
        </div>
        <Progress value={((roundIndex) / rounds.length) * 100} className="h-2" />
      </div>

      <Card className="w-full">
        <CardContent className="p-6">
          <p
            className="font-dyslexic text-lg leading-loose tracking-wide select-none"
            style={{ filter: phase === 'ready' ? 'blur(6px)' : 'none', transition: 'filter 0.3s' }}
          >
            {round.passage}
          </p>
        </CardContent>
      </Card>

      {phase === 'ready' && (
        <div className="text-center">
          <p className="font-dyslexic text-sm text-gray-500 mb-4">
            The text is blurred. Press Start when ready — it will reveal, and your timer begins.
          </p>
          <Button onClick={startReading} className="font-dyslexic">
            <ChevronRight className="mr-2 h-5 w-5" /> Start Reading
          </Button>
        </div>
      )}

      {phase === 'reading' && (
        <div className="text-center">
          <p className="font-dyslexic text-sm text-gray-500 mb-4">
            Read the passage above, then press Done when finished.
          </p>
          <Button onClick={doneReading} variant="outline" className="font-dyslexic">
            Done Reading
          </Button>
        </div>
      )}

      {phase === 'done' && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <p className="font-dyslexic text-2xl font-bold text-primary">{currentWpm} WPM</p>
          <p className="font-dyslexic text-sm text-gray-500 mt-1">Loading next passage…</p>
        </motion.div>
      )}
    </div>
  );
};

// ── Main Component ─────────────────────────────────────────────────────────────

const Assessment = () => {
  const [phase, setPhase] = useState<Phase>('intro');
  const [reversalScore, setReversalScore] = useState(0);
  const [reversalErrors, setReversalErrors] = useState<string[]>([]);
  const [avgWpm, setAvgWpm] = useState(0);
  const [questions] = useState(() => shuffle(REVERSAL_QUESTIONS));

  const handleReversalComplete = (score: number, errors: string[]) => {
    setReversalScore(score);
    setReversalErrors(errors);
    setPhase('speed');
  };

  const handleSpeedComplete = (wpm: number) => {
    setAvgWpm(wpm);
    setPhase('results');
  };

  const restart = () => {
    setPhase('intro');
    setReversalScore(0);
    setReversalErrors([]);
    setAvgWpm(0);
  };

  const accuracy = Math.round((reversalScore / REVERSAL_QUESTIONS.length) * 100);
  const { label: wpmLabel, color: wpmColor } = wpmRating(avgWpm);

  return (
    <div className="max-w-2xl mx-auto">
      <AnimatePresence mode="wait">
        {/* ── Intro ── */}
        {phase === 'intro' && (
          <motion.div
            key="intro"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="font-dyslexic flex items-center gap-2 text-2xl">
                  <Brain className="h-6 w-6 text-primary" />
                  Dyslexia Skills Assessment
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="font-dyslexic text-gray-600 leading-relaxed">
                  This quick assessment has two parts:
                </p>
                <ol className="list-decimal list-inside font-dyslexic text-gray-700 space-y-2 pl-2">
                  <li>
                    <strong>Letter Reversal Test</strong> — identify which letter or word is shown
                    ({REVERSAL_QUESTIONS.length} questions, ~2 min)
                  </li>
                  <li>
                    <strong>Reading Speed Test</strong> — read three short passages and we'll
                    measure your words per minute
                  </li>
                </ol>
                <p className="font-dyslexic text-sm text-gray-500">
                  Results are for self-awareness only and are not a medical diagnosis.
                </p>
                <Button className="w-full font-dyslexic text-lg mt-2" onClick={() => setPhase('reversal')}>
                  Begin Assessment <ChevronRight className="ml-2 h-5 w-5" />
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* ── Reversal ── */}
        {phase === 'reversal' && (
          <motion.div
            key="reversal"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="font-dyslexic text-xl">Part 1: Letter & Word Identification</CardTitle>
              </CardHeader>
              <CardContent>
                <ReversalTest questions={questions} onComplete={handleReversalComplete} />
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* ── Speed ── */}
        {phase === 'speed' && (
          <motion.div
            key="speed"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="font-dyslexic text-xl">Part 2: Reading Speed</CardTitle>
              </CardHeader>
              <CardContent>
                <SpeedTest rounds={SPEED_ROUNDS} onComplete={handleSpeedComplete} />
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* ── Results ── */}
        {phase === 'results' && (
          <motion.div
            key="results"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="font-dyslexic text-2xl text-center">Your Results</CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Score cards */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-blue-50 rounded-xl">
                    <p className="text-4xl font-bold text-blue-600">{accuracy}%</p>
                    <p className="font-dyslexic text-sm text-gray-600 mt-1">Letter Accuracy</p>
                    <Badge variant={accuracy >= 80 ? 'default' : 'destructive'} className="mt-2">
                      {accuracy >= 90 ? 'Excellent' : accuracy >= 70 ? 'Good' : 'Needs Practice'}
                    </Badge>
                  </div>
                  <div className="text-center p-4 bg-purple-50 rounded-xl">
                    <p className={`text-4xl font-bold ${wpmColor}`}>{avgWpm}</p>
                    <p className="font-dyslexic text-sm text-gray-600 mt-1">Words Per Minute</p>
                    <Badge variant="outline" className={`mt-2 ${wpmColor}`}>{wpmLabel}</Badge>
                  </div>
                </div>

                {/* Errors */}
                {reversalErrors.length > 0 && (
                  <div className="p-4 bg-orange-50 rounded-xl">
                    <p className="font-dyslexic font-semibold text-orange-800 mb-2 flex items-center gap-2">
                      <XCircle className="h-4 w-4" /> Common Confusions
                    </p>
                    <ul className="font-dyslexic text-sm text-orange-700 space-y-1 list-disc list-inside">
                      {reversalErrors.map((e, i) => <li key={i}>{e}</li>)}
                    </ul>
                  </div>
                )}

                {/* Tips based on score */}
                <div className="p-4 bg-green-50 rounded-xl">
                  <p className="font-dyslexic font-semibold text-green-800 mb-2 flex items-center gap-2">
                    <CheckCircle2 className="h-4 w-4" /> Personalised Tip
                  </p>
                  <p className="font-dyslexic text-sm text-green-700 leading-relaxed">
                    {accuracy < 70
                      ? "Practice distinguishing b/d and p/q by tracing them with your finger and saying the sound aloud. Using different colours for each can help build visual memory."
                      : avgWpm < 100
                      ? "Reading speed improves with practice. Try reading a short paragraph daily and gradually increase length. Using a ruler to track lines helps reduce skipping."
                      : "Great job! Continue practising with DyslexiNote's drawing and recognition tools to keep improving your reading and writing skills."}
                  </p>
                </div>

                <Button onClick={restart} variant="outline" className="w-full font-dyslexic">
                  <RotateCcw className="mr-2 h-4 w-4" /> Retake Assessment
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Assessment;
