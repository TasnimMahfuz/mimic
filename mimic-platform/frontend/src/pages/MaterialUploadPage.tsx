import React, { useState } from 'react';
import type { ChangeEvent } from 'react';
import { api } from '../api/client';
import { Container, Card, Button, Badge } from '../components/ui';

interface UploadStatus {
  state: 'idle' | 'uploading' | 'success' | 'error';
  message: string;
  materialId?: string;
}

export const MaterialUploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadStatus>({ state: 'idle', message: '' });
  const [uploadHistory, setUploadHistory] = useState<Array<{ id: string; name: string; time: string }>>([]);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setStatus({ state: 'idle', message: '' });
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setStatus({ state: 'error', message: 'Please select a file' });
      return;
    }

    setStatus({ state: 'uploading', message: 'Uploading and ingesting materials...' });

    try {
      const response = await api.uploadMaterial(file);
      const timestamp = new Date().toLocaleTimeString();

      setStatus({
        state: 'success',
        message: `Material uploaded successfully! ID: ${response.material_id}`,
        materialId: response.material_id,
      });

      setUploadHistory([
        {
          id: response.material_id,
          name: file.name,
          time: timestamp,
        },
        ...uploadHistory,
      ]);

      setFile(null);
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Upload failed';
      setStatus({
        state: 'error',
        message: `Error: ${errorMsg}`,
      });
    }
  };

  return (
    <Container>
      <div className="max-w-2xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-neutral-900">Upload Materials</h1>
          <p className="text-neutral-600 mt-2">Upload text files or documents for RAG ingestion</p>
        </div>

        <Card className="mb-8">
          <div className="space-y-6">
            {/* File Upload Area */}
            <div>
              <label className="block text-sm font-medium text-neutral-700 mb-4">Select File</label>
              <div className="border-2 border-dashed border-neutral-300 rounded-lg p-8 text-center hover:border-primary transition-colors">
                <svg
                  className="mx-auto h-12 w-12 text-neutral-400 mb-3"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <p className="text-neutral-600 mb-2">Drag and drop your file here, or click to select</p>
                <p className="text-xs text-neutral-500 mb-4">Supported formats: .txt, .pdf, .md</p>
                <input
                  type="file"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-input"
                  accept=".txt,.pdf,.md"
                />
                <label htmlFor="file-input">
                  <Button type="button" variant="outline" size="sm" onClick={() => document.getElementById('file-input')?.click()}>
                    Choose File
                  </Button>
                </label>
              </div>

              {file && (
                <div className="mt-4 p-4 bg-neutral-50 rounded-lg flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <svg
                      className="h-6 w-6 text-neutral-400"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    <div>
                      <p className="font-medium text-neutral-900">{file.name}</p>
                      <p className="text-xs text-neutral-500">{(file.size / 1024).toFixed(2)} KB</p>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFile(null)}
                    className="text-neutral-400 hover:text-neutral-600"
                  >
                    ✕
                  </button>
                </div>
              )}
            </div>

            {/* Status Messages */}
            {status.state !== 'idle' && (
              <div
                className={`p-4 rounded-lg ${
                  status.state === 'uploading' ? 'bg-blue-50 text-blue-800' : status.state === 'success' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-800'
                }`}
              >
                <p className="text-sm">{status.message}</p>
              </div>
            )}

            {/* Upload Button */}
            <Button
              onClick={handleUpload}
              variant="primary"
              size="lg"
              isLoading={status.state === 'uploading'}
              disabled={!file || status.state === 'uploading'}
              className="w-full"
            >
              Upload & Ingest Material
            </Button>
          </div>
        </Card>

        {/* Upload History */}
        {uploadHistory.length > 0 && (
          <Card>
            <h2 className="text-lg font-semibold text-neutral-900 mb-4">Recent Uploads</h2>
            <div className="space-y-3">
              {uploadHistory.map((item) => (
                <div key={item.id} className="flex items-center justify-between p-4 bg-neutral-50 rounded-lg hover:bg-neutral-100">
                  <div>
                    <p className="font-medium text-neutral-900">{item.name}</p>
                    <p className="text-xs text-neutral-500">{item.time}</p>
                  </div>
                  <Badge variant="success">ID: {item.id}</Badge>
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>
    </Container>
  );
};
