import { useState } from 'react';
import { cn } from '@/lib/utils';

interface ColorPickerProps {
  onColorChange: (color: string) => void;
  defaultColor?: string;
}

const ColorPicker = ({ onColorChange, defaultColor = '#000000' }: ColorPickerProps) => {
  const [activeColor, setActiveColor] = useState(defaultColor);
  
  const colors = [
    { color: '#000000', label: 'Black' },
    { color: '#2D9CDB', label: 'Blue' },
    { color: '#6FCF97', label: 'Green' },
    { color: '#BB6BD9', label: 'Purple' },
    { color: '#F2994A', label: 'Orange' },
    { color: '#EB5757', label: 'Red' }
  ];

  const handleColorClick = (color: string) => {
    setActiveColor(color);
    onColorChange(color);
  };

  return (
    <div className="flex items-center space-x-2">
      {colors.map(({ color, label }) => (
        <button
          key={color}
          onClick={() => handleColorClick(color)}
          className={cn(
            "color-swatch",
            activeColor === color && "active"
          )}
          style={{ backgroundColor: color }}
          title={label}
          aria-label={`${label} color`}
        />
      ))}
    </div>
  );
};

export default ColorPicker;
