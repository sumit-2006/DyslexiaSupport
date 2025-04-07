import { useState } from 'react';
import { cn } from '@/lib/utils';

interface PenSizePickerProps {
  onSizeChange: (size: number) => void;
  defaultSize?: number;
}

const PenSizePicker = ({ onSizeChange, defaultSize = 2 }: PenSizePickerProps) => {
  const [activeSize, setActiveSize] = useState(defaultSize);
  
  const sizes = [
    { size: 2, label: 'Extra Small' },
    { size: 4, label: 'Small' },
    { size: 6, label: 'Medium' },
    { size: 8, label: 'Large' }
  ];

  const handleSizeClick = (size: number) => {
    setActiveSize(size);
    onSizeChange(size);
  };

  return (
    <div className="flex space-x-2">
      {sizes.map(({ size, label }) => (
        <button
          key={size}
          onClick={() => handleSizeClick(size)}
          className={cn(
            "pen-size p-2",
            activeSize === size && "active"
          )}
          title={label}
          aria-label={`${label} pen size`}
        >
          <div
            className="pen-size-circle"
            style={{ width: `${size}px`, height: `${size}px` }}
          />
        </button>
      ))}
    </div>
  );
};

export default PenSizePicker;
