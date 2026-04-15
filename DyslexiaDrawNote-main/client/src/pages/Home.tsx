import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import NoteCard from '@/components/NoteCard';
import { Search, Plus, BookOpen, Star, FileText, Lightbulb, Activity } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';
import type { Note } from '@shared/schema';

type Tab = 'all' | 'recent' | 'favorites';

const DYSLEXIA_TIPS = [
  "Try using a cream or light yellow background instead of white — it reduces glare and helps reading.",
  "Reading in short bursts with breaks helps more than long reading sessions.",
  "Text-to-speech tools can help you catch errors that your eyes might skip over.",
  "Using a finger or ruler to track lines while reading can significantly reduce skipping.",
  "Larger fonts and wider line spacing make text much easier to read.",
  "Breaking words into syllables (syl-la-bles) makes them easier to decode.",
  "Mind maps are often easier to create and read than linear notes.",
  "Recording lectures and replaying them at 0.75x speed can help with comprehension.",
  "Color-coding notes by topic engages spatial memory and aids recall.",
  "Spaced repetition — reviewing material at increasing intervals — helps long-term retention.",
];

function getTipOfTheDay(): string {
  const dayIndex = new Date().getDate() % DYSLEXIA_TIPS.length;
  return DYSLEXIA_TIPS[dayIndex];
}

interface Stats {
  totalNotes: number;
  favoriteNotes: number;
  totalWords: number;
  recentNotes: number;
}

const StatCard = ({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: number | string;
  color: string;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 16 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.35 }}
  >
    <Card className={`border-l-4 ${color}`}>
      <CardContent className="p-4 flex items-center gap-4">
        <div className={`p-2 rounded-full bg-opacity-10 ${color.replace('border-l-', 'bg-')}`}>
          <Icon className="h-5 w-5 text-current" />
        </div>
        <div>
          <p className="text-2xl font-bold font-dyslexic">{value}</p>
          <p className="text-sm text-gray-500 font-dyslexic">{label}</p>
        </div>
      </CardContent>
    </Card>
  </motion.div>
);

const Home = () => {
  const [, navigate] = useLocation();
  const [activeTab, setActiveTab] = useState<Tab>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const { data: notes, isLoading, error } = useQuery<Note[]>({
    queryKey: ['/api/notes'],
  });

  const { data: stats } = useQuery<Stats>({
    queryKey: ['/api/stats'],
  });

  const getFilteredNotes = () => {
    if (!notes) return [];
    let filtered = [...notes];

    if (activeTab === 'recent') {
      filtered = [...filtered]
        .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime())
        .slice(0, 10);
    } else if (activeTab === 'favorites') {
      filtered = filtered.filter(note => note.isFavorite);
    }

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(note =>
        note.title.toLowerCase().includes(query) ||
        (note.recognizedText && note.recognizedText.toLowerCase().includes(query))
      );
    }

    return filtered;
  };

  return (
    <div>
      {/* Welcome banner */}
      <motion.div
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="mb-8 p-5 bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl border border-blue-100"
      >
        <h2 className="text-2xl font-bold font-dyslexic text-primary mb-1">
          Welcome to DyslexiNote 📝
        </h2>
        <p className="text-gray-600 font-dyslexic text-sm leading-relaxed">
          A dyslexia-friendly note-taking app with AI-powered handwriting recognition.
        </p>
      </motion.div>

      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          icon={FileText}
          label="Total Notes"
          value={stats?.totalNotes ?? (notes?.length ?? '—')}
          color="border-l-blue-400"
        />
        <StatCard
          icon={Star}
          label="Favorites"
          value={stats?.favoriteNotes ?? '—'}
          color="border-l-yellow-400"
        />
        <StatCard
          icon={BookOpen}
          label="Words Written"
          value={stats?.totalWords ?? '—'}
          color="border-l-green-400"
        />
        <StatCard
          icon={Activity}
          label="Last 7 Days"
          value={stats?.recentNotes ?? '—'}
          color="border-l-purple-400"
        />
      </div>

      {/* Tip of the day */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="mb-8 p-4 bg-amber-50 border border-amber-200 rounded-xl flex gap-3 items-start"
      >
        <Lightbulb className="h-5 w-5 text-amber-500 mt-0.5 shrink-0" />
        <div>
          <p className="font-dyslexic font-semibold text-amber-800 text-sm mb-0.5">Tip of the Day</p>
          <p className="font-dyslexic text-amber-700 text-sm leading-relaxed">{getTipOfTheDay()}</p>
        </div>
      </motion.div>

      {/* Notes header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6">
        <h2 className="text-xl font-bold mb-4 md:mb-0 font-dyslexic">My Notes</h2>

        <div className="w-full md:w-auto flex items-center space-x-3">
          <div className="relative w-full md:w-64">
            <Input
              type="text"
              placeholder="Search notes..."
              className="pl-10 pr-4 py-3 font-dyslexic w-full"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search className="text-gray-400 h-5 w-5" />
            </div>
          </div>

          <Button
            onClick={() => navigate('/note')}
            className="bg-secondary text-white font-dyslexic font-semibold"
          >
            <Plus className="mr-2 h-5 w-5" />
            New Note
          </Button>
        </div>
      </div>

      {/* Tab filter */}
      <div className="flex space-x-1 mb-6 border-b border-gray-200">
        {(['all', 'recent', 'favorites'] as Tab[]).map(tab => (
          <Button
            key={tab}
            variant="ghost"
            className={`py-3 px-5 rounded-none font-dyslexic font-semibold text-base capitalize ${
              activeTab === tab
                ? 'border-b-2 border-primary text-primary'
                : 'text-gray-500 hover:text-primary'
            }`}
            onClick={() => setActiveTab(tab)}
          >
            {tab === 'all' ? 'All Notes' : tab === 'recent' ? 'Recent' : 'Favorites'}
          </Button>
        ))}
      </div>

      {/* Notes grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {isLoading ? (
          Array.from({ length: 8 }).map((_, index) => (
            <div key={index} className="bg-white rounded-lg shadow-md overflow-hidden">
              <Skeleton className="h-40 w-full" />
              <div className="p-4">
                <Skeleton className="h-6 w-3/4 mb-2" />
                <Skeleton className="h-4 w-full mb-1" />
                <Skeleton className="h-4 w-2/3 mb-3" />
                <div className="flex justify-between items-center">
                  <Skeleton className="h-4 w-20" />
                  <div className="flex space-x-2">
                    <Skeleton className="h-8 w-8 rounded-full" />
                    <Skeleton className="h-8 w-8 rounded-full" />
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : error ? (
          <div className="col-span-full p-8 text-center">
            <p className="font-dyslexic text-lg text-red-500">
              Error loading notes. Please try again later.
            </p>
          </div>
        ) : getFilteredNotes().length === 0 ? (
          <div className="col-span-full p-8 text-center">
            <p className="font-dyslexic text-lg text-gray-500">
              {searchQuery
                ? `No notes found matching "${searchQuery}"`
                : activeTab === 'favorites'
                  ? "You don't have any favorite notes yet"
                  : "You don't have any notes yet. Create one!"}
            </p>
            <Button
              onClick={() => navigate('/note')}
              className="mt-4 font-dyslexic"
            >
              <Plus className="mr-2 h-5 w-5" />
              Create New Note
            </Button>
          </div>
        ) : (
          getFilteredNotes().map(note => (
            <motion.div
              key={note.id}
              initial={{ opacity: 0, scale: 0.97 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.2 }}
            >
              <NoteCard note={note} />
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
};

export default Home;

