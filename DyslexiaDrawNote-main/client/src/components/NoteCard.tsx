import { useState } from 'react';
import { useLocation } from 'wouter';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Star, Edit, Trash2 } from 'lucide-react';
import { formatDate } from '@/lib/utils';
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { apiRequest } from '@/lib/queryClient';
import { queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import type { Note } from '@shared/schema';

interface NoteCardProps {
  note: Note;
}

const NoteCard = ({ note }: NoteCardProps) => {
  const [, navigate] = useLocation();
  const { toast } = useToast();
  
  const [isFavorite, setIsFavorite] = useState(note.isFavorite);
  const [isDeleting, setIsDeleting] = useState(false);
  
  const handleEditClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigate(`/note/${note.id}`);
  };
  
  const handleCardClick = () => {
    navigate(`/note/${note.id}`);
  };
  
  const toggleFavorite = async (e: React.MouseEvent) => {
    e.stopPropagation();
    
    try {
      setIsFavorite(!isFavorite);
      
      await apiRequest('PATCH', `/api/notes/${note.id}/favorite`, {
        isFavorite: !isFavorite
      });
      
      queryClient.invalidateQueries({ queryKey: ['/api/notes'] });
      
      toast({
        title: !isFavorite ? "Added to favorites" : "Removed from favorites",
        description: `"${note.title}" ${!isFavorite ? 'added to' : 'removed from'} favorites.`,
      });
    } catch (error) {
      // Revert on error
      setIsFavorite(isFavorite);
      
      toast({
        title: "Error updating favorite status",
        description: "Please try again later.",
        variant: "destructive",
      });
    }
  };
  
  const handleDelete = async () => {
    setIsDeleting(true);
    
    try {
      await apiRequest('DELETE', `/api/notes/${note.id}`);
      
      queryClient.invalidateQueries({ queryKey: ['/api/notes'] });
      
      toast({
        title: "Note deleted",
        description: `"${note.title}" has been deleted.`,
      });
    } catch (error) {
      toast({
        title: "Error deleting note",
        description: "Please try again later.",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
    }
  };

  return (
    <Card 
      className="overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
      onClick={handleCardClick}
    >
      <div className="h-40 bg-gray-100 relative">
        {note.preview ? (
          <img 
            src={note.preview} 
            alt={`Preview of ${note.title}`} 
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            No preview
          </div>
        )}
        <div className="absolute top-2 right-2">
          <Button
            variant="outline"
            size="icon"
            className={`w-8 h-8 rounded-full bg-white ${isFavorite ? 'text-accent' : 'text-gray-600 hover:text-accent'}`}
            onClick={toggleFavorite}
          >
            <Star className="h-4 w-4" fill={isFavorite ? 'currentColor' : 'none'} />
          </Button>
        </div>
      </div>
      
      <CardContent className="p-4">
        <h3 className="font-dyslexic text-xl font-bold mb-2 line-clamp-1">
          {note.title || 'Untitled Note'}
        </h3>
        <p className="font-dyslexic text-gray-600 text-sm mb-3 line-clamp-2">
          {note.recognizedText || 'No text content'}
        </p>
      </CardContent>
      
      <CardFooter className="px-4 pb-4 pt-0 flex justify-between items-center">
        <span className="font-dyslexic text-sm text-gray-500">
          {formatDate(note.createdAt)}
        </span>
        <div className="flex space-x-2">
          <Button
            variant="ghost"
            size="icon"
            className="w-8 h-8 rounded-full hover:text-primary"
            onClick={handleEditClick}
          >
            <Edit className="h-4 w-4" />
          </Button>
          
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="w-8 h-8 rounded-full hover:text-red-500"
                onClick={(e) => e.stopPropagation()}
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle className="font-dyslexic">Delete Note</AlertDialogTitle>
                <AlertDialogDescription className="font-dyslexic">
                  Are you sure you want to delete "{note.title}"? This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel onClick={(e) => e.stopPropagation()}>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDelete();
                  }}
                  className="bg-red-500 hover:bg-red-600 focus:ring-red-500"
                  disabled={isDeleting}
                >
                  {isDeleting ? 'Deleting...' : 'Delete'}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </CardFooter>
    </Card>
  );
};

export default NoteCard;
