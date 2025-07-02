import { Link } from "wouter";

const Footer = () => {
  return (
    <footer className="bg-white py-4 px-6 mt-8 border-t border-gray-200">
      <div className="container mx-auto">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <p className="font-dyslexic text-gray-600 mb-4 md:mb-0">
            © {new Date().getFullYear()} DyslexiNote - Accessible Note-Taking App
          </p>
          <div className="flex items-center space-x-4">
            <Link href="/help">
              <span className="text-gray-600 hover:text-primary transition font-inter cursor-pointer">
                Help
              </span>
            </Link>
            <Link href="/privacy">
              <span className="text-gray-600 hover:text-primary transition font-inter cursor-pointer">
                Privacy
              </span>
            </Link>
            <Link href="/terms">
              <span className="text-gray-600 hover:text-primary transition font-inter cursor-pointer">
                Terms
              </span>
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
