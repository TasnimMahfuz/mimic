import React, { useState, useRef, useEffect } from 'react';
import type { FormEvent } from 'react';
import { api } from '../api/client';
import { Container, Card, Button, Input, Spinner } from '../components/ui';
import { DebugPanel } from '../components/DebugPanel';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  context?: string;
  debugResponse?: unknown;
  latency?: number;
}

export const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedMessage, setExpandedMessage] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    console.log('Adding user message:', userMessage);
    setMessages((prev) => {
      const newMessages = [...prev, userMessage];
      console.log('Messages after adding user:', newMessages);
      return newMessages;
    });
    setInput('');
    setError(null);
    setIsLoading(true);

    const startTime = Date.now();

    try {
      const response = await api.chatQuery(input);
      const latency = Date.now() - startTime;

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response,
        context: response.context,
        debugResponse: response,
        latency,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to get response';
      setError(errorMsg);

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error: ${errorMsg}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container className="flex flex-col h-screen pb-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-neutral-900">Chat Interface</h1>
        <p className="text-neutral-600 mt-2">Ask questions about uploaded materials</p>
      </div>

      <div className="flex-1 overflow-y-auto mb-6 space-y-4">
        {console.log('Rendering messages, count:', messages.length)}
        {messages.length === 0 ? (
          <Card className="h-full flex items-center justify-center text-center">
            <div>
              <svg className="h-16 w-16 text-neutral-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M12 8v4m0 4v.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p className="text-neutral-600 mb-2">Start a conversation</p>
              <p className="text-sm text-neutral-500">Upload materials first, then ask questions about them</p>
            </div>
          </Card>
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-xl ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                  <div
                    className={`rounded-lg p-6 shadow-md border cursor-pointer hover:shadow-lg transition-shadow ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-white border-gray-200 text-gray-900'
                    }`}
                    onClick={() =>
                      message.debugResponse && setExpandedMessage(expandedMessage === message.id ? null : message.id)
                    }
                  >
                    <p className={`text-sm ${message.role === 'user' ? 'text-blue-100' : 'text-gray-600'}`}>
                      {message.timestamp}
                    </p>
                    <p className={`mt-2 whitespace-pre-wrap break-words`}>
                      {String(message.content)}
                    </p>
                  </div>

                  {/* Context Panel for Assistant Messages */}
                  {message.role === 'assistant' && message.context ? (
                    <div className="mt-2">
                      <button
                        onClick={() => setExpandedMessage(expandedMessage === `${message.id}-context` ? null : `${message.id}-context`)}
                        className="text-xs text-primary hover:underline font-medium"
                      >
                        {expandedMessage === `${message.id}-context` ? '↓ Hide' : '→ Show'} RAG Context
                      </button>

                      {expandedMessage === `${message.id}-context` ? (
                        <Card className="mt-2 bg-amber-50 border-amber-200 p-4">
                          <p className="text-xs font-medium text-amber-900 mb-2">Retrieved Context:</p>
                          <p className="text-xs text-amber-800 whitespace-pre-wrap break-words">{message.context}</p>
                        </Card>
                      ) : null}
                    </div>
                  ) : null}

                  {/* Debug Panel for Assistant Messages */}
                  {message.role === 'assistant' && message.debugResponse && expandedMessage === message.id ? (
                    <div className="mt-4">
                      <DebugPanel response={message.debugResponse} startTime={message.latency ? Date.now() - message.latency : undefined} />
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </>
        )}

        {isLoading && (
          <div className="flex justify-start">
            <Card className="flex items-center gap-3">
              <Spinner />
              <span className="text-neutral-600">Thinking...</span>
            </Card>
          </div>
        )}

        <div ref={scrollRef} />
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-red-50 text-red-800 rounded-lg mb-4 flex items-center justify-between">
          <span className="text-sm">{error}</span>
          <button onClick={() => setError(null)} className="text-red-600 hover:text-red-800">
            ✕
          </button>
        </div>
      )}

      {/* Input Form */}
      <form onSubmit={handleSendMessage} className="flex gap-3">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isLoading}
          className="flex-1"
        />
        <Button
          type="submit"
          variant="primary"
          size="lg"
          isLoading={isLoading}
          disabled={!input.trim() || isLoading}
        >
          Send
        </Button>
      </form>
    </Container>
  );
};
