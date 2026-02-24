import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../api/auth';
import { Button } from './ui';

export const Navigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout, isAuthenticated } = useAuth();

  if (!isAuthenticated) return null;

  const isActive = (path: string) => location.pathname === path;

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="bg-white border-b border-neutral-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <button
            onClick={() => navigate('/dashboard')}
            className="flex items-center gap-2 hover:opacity-80 transition-opacity"
          >
            <div className="h-8 w-8 bg-primary rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-lg">M</span>
            </div>
            <span className="font-bold text-neutral-900 text-lg">MIMIC</span>
          </button>

          {/* Navigation Links */}
          <div className="flex items-center gap-1">
            <NavLink
              to="/chat"
              label="Chat"
              isActive={isActive('/chat')}
            />
            <NavLink
              to="/materials"
              label="Materials"
              isActive={isActive('/materials')}
            />
            <NavLink
              to="/mimic"
              label="MIMIC Analysis"
              isActive={isActive('/mimic')}
            />
            <NavLink
              to="/test-panel"
              label="Test Panel"
              isActive={isActive('/test-panel')}
            />
          </div>

          {/* Logout Button */}
          <Button onClick={handleLogout} variant="ghost" size="sm">
            Logout
          </Button>
        </div>
      </div>
    </nav>
  );
};

interface NavLinkProps {
  to: string;
  label: string;
  isActive: boolean;
}

const NavLink: React.FC<NavLinkProps> = ({ to, label, isActive }) => {
  const navigate = useNavigate();

  return (
    <button
      onClick={() => navigate(to)}
      className={`px-4 py-2 rounded-lg font-medium transition-colors ${
        isActive
          ? 'bg-primary text-white'
          : 'text-neutral-600 hover:text-neutral-900 hover:bg-neutral-100'
      }`}
    >
      {label}
    </button>
  );
};
