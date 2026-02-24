import React, { useState } from 'react';
import type { ChangeEvent } from 'react';
import { api } from '../api/client';
import { Container, Card, Button, Input, Badge } from '../components/ui';

interface TestResult {
  endpoint: string;
  method: string;
  status: 'idle' | 'loading' | 'success' | 'error';
  statusCode?: number;
  response: unknown;
  error?: string;
  duration?: number;
}

export const TestPanelPage: React.FC = () => {
  const [results, setResults] = useState<TestResult[]>([]);
  const [queryInput, setQueryInput] = useState('');
  const [fileToUpload, setFileToUpload] = useState<File | null>(null);

  const addResult = (result: TestResult) => {
    setResults((prev) => [result, ...prev.slice(0, 4)]);
  };

  const testEndpoint = async (endpoint: string, method: string = 'GET', data?: unknown) => {
    const result: TestResult = {
      endpoint,
      method,
      status: 'loading',
      response: null,
    };
    addResult(result);

    const startTime = Date.now();
    try {
      const response = await api.request(method, endpoint, data);
      const duration = Date.now() - startTime;

      addResult({
        endpoint,
        method,
        status: 'success',
        statusCode: response.status,
        response: response.data,
        duration,
      });
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMsg = error instanceof Error ? error.message : 'Unknown error';

      addResult({
        endpoint,
        method,
        status: 'error',
        error: errorMsg,
        response: error instanceof Error && 'response' in error ? (error as any).response?.data : null,
        duration,
      });
    }
  };

  const handleChatQuery = () => {
    if (queryInput.trim()) {
      testEndpoint('/chat/query', 'POST', { query: queryInput });
    }
  };

  const handleMaterialUpload = () => {
    if (fileToUpload) {
      testEndpoint('/materials/upload', 'POST', fileToUpload);
    }
  };

  return (
    <Container>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-neutral-900">API Test Panel</h1>
          <p className="text-neutral-600 mt-2">Manually test backend endpoints</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Chat Query Test */}
          <Card>
            <h2 className="text-lg font-semibold text-neutral-900 mb-4">Test Chat Query</h2>
            <div className="space-y-3">
              <Input
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                placeholder="Enter a test query"
              />
              <Button
                onClick={handleChatQuery}
                variant="primary"
                size="md"
                className="w-full"
              >
                Send Query
              </Button>
            </div>
          </Card>

          {/* Material Upload Test */}
          <Card>
            <h2 className="text-lg font-semibold text-neutral-900 mb-4">Test Material Upload</h2>
            <div className="space-y-3">
              <div className="border-2 border-dashed border-neutral-300 rounded-lg p-4 text-center">
                <input
                  type="file"
                  onChange={(e: ChangeEvent<HTMLInputElement>) => setFileToUpload(e.target.files?.[0] || null)}
                  className="hidden"
                  id="test-file-input"
                />
                <label htmlFor="test-file-input" className="cursor-pointer">
                  {fileToUpload ? (
                    <p className="text-sm text-neutral-700">{fileToUpload.name}</p>
                  ) : (
                    <p className="text-sm text-neutral-500">Click to select file</p>
                  )}
                </label>
              </div>
              <Button
                onClick={handleMaterialUpload}
                variant="primary"
                size="md"
                disabled={!fileToUpload}
                className="w-full"
              >
                Upload
              </Button>
            </div>
          </Card>
        </div>

        {/* Test Results */}
        {results.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold text-neutral-900 mb-4">Recent Test Results</h2>
            <div className="space-y-4">
              {results.map((result, idx) => (
                <Card key={idx} className="border-neutral-200">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <Badge
                        variant={
                          result.status === 'success'
                            ? 'success'
                            : result.status === 'error'
                              ? 'error'
                              : 'default'
                        }
                      >
                        {result.method}
                      </Badge>
                      <div>
                        <p className="font-mono text-sm text-neutral-900">{result.endpoint}</p>
                        {result.statusCode && (
                          <p className="text-xs text-neutral-500">Status: {result.statusCode}</p>
                        )}
                        {result.duration && (
                          <p className="text-xs text-neutral-500">Duration: {result.duration}ms</p>
                        )}
                      </div>
                    </div>
                    <div>
                      {result.status === 'loading' && (
                        <Badge variant="default">Loading...</Badge>
                      )}
                      {result.status === 'success' && (
                        <Badge variant="success">Success</Badge>
                      )}
                      {result.status === 'error' && (
                        <Badge variant="error">Error</Badge>
                      )}
                    </div>
                  </div>

                  {result.error && (
                    <div className="p-3 bg-red-50 rounded mb-3 text-sm text-red-700">
                      {result.error}
                    </div>
                  )}

                  <div className="bg-neutral-900 rounded p-3 overflow-auto max-h-48">
                    <pre className="text-xs text-neutral-300 font-mono">
                      {JSON.stringify(result.response || result.error, null, 2)}
                    </pre>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Quick Test Buttons */}
        <Card className="mt-8">
          <h2 className="text-lg font-semibold text-neutral-900 mb-4">Quick Tests</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <Button
              onClick={() => testEndpoint('/')}
              variant="outline"
              size="sm"
              className="w-full"
            >
              GET /
            </Button>
            <Button
              onClick={() => testEndpoint('/auth/register', 'POST', { email: 'test@test.com', password: 'test', role: 'student' })}
              variant="outline"
              size="sm"
              className="w-full"
            >
              POST /auth/register
            </Button>
            <Button
              onClick={() => testEndpoint('/chat/query', 'POST', { query: 'test' })}
              variant="outline"
              size="sm"
              className="w-full"
            >
              POST /chat/query
            </Button>
          </div>
        </Card>
      </div>
    </Container>
  );
};
