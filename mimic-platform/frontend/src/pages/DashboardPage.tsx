import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../api/auth';
import { Container, Card, Button } from '../components/ui';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { logout } = useAuth();

  const features = [
    {
      title: 'Chat Interface',
      description: 'Ask questions about uploaded materials with RAG context display',
      icon: '💬',
      action: () => navigate('/chat'),
    },
    {
      title: 'Material Upload',
      description: 'Upload text files and documents for ingestion into the RAG system',
      icon: '📤',
      action: () => navigate('/materials'),
    },
    {
      title: 'Test Panel',
      description: 'Manual API testing panel for backend development and debugging',
      icon: '🧪',
      action: () => navigate('/test-panel'),
    },
  ];

  return (
    <Container>
      <div className="py-12">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-neutral-900 mb-4">Welcome to MIMIC</h1>
          <p className="text-lg text-neutral-600">Testing Platform for RAG-based Chat Application</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {features.map((feature, idx) => (
            <Card
              key={idx}
              hover
              className="flex flex-col items-center text-center"
              onClick={feature.action}
            >
              <div className="text-5xl mb-4">{feature.icon}</div>
              <h2 className="text-xl font-semibold text-neutral-900 mb-2">{feature.title}</h2>
              <p className="text-neutral-600 text-sm mb-6 flex-1">{feature.description}</p>
              <Button variant="primary" size="sm">
                Open
              </Button>
            </Card>
          ))}
        </div>

        <Card className="bg-gradient-to-br from-purple-50 to-neutral-50 border-purple-200 p-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-semibold text-neutral-900 mb-2">🚀 Quick Start</h3>
              <ol className="text-sm text-neutral-600 space-y-2">
                <li>1. Upload materials on the Materials page</li>
                <li>2. Wait for ingestion to complete</li>
                <li>3. Ask questions on the Chat page</li>
              </ol>
            </div>
            <div>
              <h3 className="font-semibold text-neutral-900 mb-2">🔧 Testing</h3>
              <ul className="text-sm text-neutral-600 space-y-2">
                <li>• Debug Panel shows RAG chunks</li>
                <li>• Similarity scores displayed</li>
                <li>• Request latency tracked</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-neutral-900 mb-2">ℹ️ Features</h3>
              <ul className="text-sm text-neutral-600 space-y-2">
                <li>• Real-time chat interface</li>
                <li>• RAG context display</li>
                <li>• API test panel</li>
              </ul>
            </div>
          </div>
        </Card>

        <div className="mt-8 text-center">
          <Button onClick={() => logout()} variant="ghost" size="md">
            Logout
          </Button>
        </div>
      </div>
    </Container>
  );
};
