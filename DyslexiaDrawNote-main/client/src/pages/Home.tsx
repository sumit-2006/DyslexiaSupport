import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useLocation } from 'wouter';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import NoteCard from '@/components/NoteCard';
import { Search, Plus } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';
import type { Note } from '@shared/schema';

type Tab = 'all' | 'recent' | 'favorites';

const Home = () => {
  const [, navigate] = useLocation();
  const [activeTab, setActiveTab] = useState<Tab>('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  const { data: notes, isLoading, error } = useQuery<Note[]>({
    queryKey: ['/api/notes'],
  });

  // Filter notes based on active tab and search query
  const getFilteredNotes = () => {
    if (!notes) return [];
    
    let filtered = [...notes];
    
    // Apply tab filter
    if (activeTab === 'recent') {
      // Sort by date descending and take first 10
      filtered = [...filtered].sort((a, b) => 
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
      ).slice(0, 10);
    } else if (activeTab === 'favorites') {
      filtered = filtered.filter(note => note.isFavorite);
    }
    
    // Apply search filter
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
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <h2 className="text-2xl font-bold mb-4 md:mb-0 font-dyslexic">My Notes</h2>
        
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
            Create New
          </Button>
        </div>
      </div>
      
      {/* Notes Organization Section */}
      <div className="mb-8">
        <div className="flex space-x-1 mb-4 border-b border-gray-200">
          <Button
            variant="ghost"
            className={`py-3 px-5 rounded-none font-dyslexic font-semibold text-lg ${
              activeTab === 'all' 
                ? 'border-b-2 border-primary text-primary' 
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('all')}
          >
            All Notes
          </Button>
          
          <Button
            variant="ghost" 
            className={`py-3 px-5 rounded-none font-dyslexic font-semibold text-lg ${
              activeTab === 'recent' 
                ? 'border-b-2 border-primary text-primary' 
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('recent')}
          >
            Recent
          </Button>
          
          <Button
            variant="ghost"
            className={`py-3 px-5 rounded-none font-dyslexic font-semibold text-lg ${
              activeTab === 'favorites' 
                ? 'border-b-2 border-primary text-primary' 
                : 'text-gray-600'
            }`}
            onClick={() => setActiveTab('favorites')}
          >
            Favorites
          </Button>
        </div>
      </div>

      {/* Notes Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {isLoading ? (
          // Loading skeleton
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
            <NoteCard key={note.id} note={note} />
          ))
        )}
      </div>
    </div>
  );
};

export default Home;
