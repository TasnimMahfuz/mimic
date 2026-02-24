import React, { useState } from 'react';
import type { FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../api/auth';
import { Container, Card, Button, Input } from '../components/ui';

export const LoginPage: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState<Record<string, string>>({});
  const { login, isLoading, error } = useAuth();
  const navigate = useNavigate();

  const validateForm = () => {
    const newErrors: Record<string, string> = {};
    if (!email) newErrors.email = 'Email is required';
    if (!password) newErrors.password = 'Password is required';
    if (email && !email.includes('@')) newErrors.email = 'Enter a valid email';
    return newErrors;
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const newErrors = validateForm();

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    try {
      await login(email, password);
      navigate('/dashboard');
    } catch {
      setErrors({ submit: error || 'Login failed' });
    }
  };

  return (
    <Container className="flex items-center justify-center min-h-screen">
      <Card className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-neutral-900">MIMIC</h1>
          <p className="text-neutral-600 mt-2">Test Platform Login</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            label="Email"
            type="email"
            placeholder="your@email.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            error={errors.email}
          />

          <Input
            label="Password"
            type="password"
            placeholder="••••••••"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            error={errors.password}
          />

          {errors.submit && <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">{errors.submit}</div>}

          <Button type="submit" variant="primary" size="lg" isLoading={isLoading} className="w-full">
            Sign In
          </Button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-neutral-600">
            Don't have an account?{' '}
            <button onClick={() => navigate('/register')} className="text-primary font-medium hover:underline">
              Sign up
            </button>
          </p>
        </div>

        <div className="mt-8 pt-6 border-t border-neutral-200">
          <p className="text-xs text-neutral-500 text-center mb-3">Demo Credentials</p>
          <div className="space-y-2 text-xs">
            <p className="text-neutral-600">
              <strong>Student:</strong> student@test.com / password
            </p>
            <p className="text-neutral-600">
              <strong>Teacher:</strong> teacher@test.com / password
            </p>
          </div>
        </div>
      </Card>
    </Container>
  );
};
