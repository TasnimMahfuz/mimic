import React, { useState } from 'react';
import { Card, Badge } from './ui';

interface DebugPanelProps {
  response: unknown;
  startTime?: number;
}

export const DebugPanel: React.FC<DebugPanelProps> = ({ response, startTime }) => {
  const [expanded, setExpanded] = useState(false);

  const latency = startTime ? Math.round((Date.now() - startTime) * 100) / 100 : null;

  // Parse response to extract chunks if available
  const getChunks = (resp: unknown): Array<{ chunk: string; score: number }> => {
    if (!resp || typeof resp !== 'object') return [];

    const r = resp as Record<string, unknown>;

    // Try to extract context string with chunk markers
    if (r.context && typeof r.context === 'string') {
      const chunks: Array<{ chunk: string; score: number }> = [];
      const lines = r.context.split('\n');

      let currentChunk = '';
      let currentScore = 0;

      lines.forEach((line: string) => {
        const scoreMatch = line.match(/\[Similarity: ([\d.]+)\]/);
        if (scoreMatch) {
          currentScore = parseFloat(scoreMatch[1]);
          currentChunk = line.replace(/\[Similarity: [\d.]+\]/, '').trim();
        } else if (line.trim() && currentChunk) {
          currentChunk += ' ' + line.trim();
        }

        if (currentChunk && currentScore > 0) {
          chunks.push({ chunk: currentChunk, score: currentScore });
          currentChunk = '';
          currentScore = 0;
        }
      });

      return chunks;
    }

    return [];
  };

  const chunks = getChunks(response);
  const chunkCount = chunks.length;

  return (
    <div className="mt-6 border-t border-neutral-200 pt-6">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center justify-between w-full p-4 bg-neutral-50 hover:bg-neutral-100 rounded-lg transition-colors"
      >
        <div className="flex items-center gap-3">
          <svg
            className={`h-5 w-5 text-neutral-600 transition-transform ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
          <span className="font-medium text-neutral-900">Debug Information</span>
        </div>
        <div className="flex items-center gap-2">
          {latency && <Badge variant="default">{latency}ms</Badge>}
          <Badge variant="default">{chunkCount} chunks</Badge>
        </div>
      </button>

      {expanded && (
        <div className="mt-4 space-y-6">
          {/* Latency */}
          {latency !== null && (
            <Card className="bg-blue-50 border-blue-200">
              <h3 className="font-semibold text-neutral-900 mb-2">Request Latency</h3>
              <p className="text-2xl font-bold text-primary">{latency}ms</p>
            </Card>
          )}

          {/* Retrieved Chunks */}
          {chunkCount > 0 ? (
            <Card className="bg-purple-50 border-purple-200">
              <h3 className="font-semibold text-neutral-900 mb-4">Retrieved Chunks ({chunkCount})</h3>
              <div className="space-y-3">
                {chunks.map((item, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-white rounded border border-purple-200 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <span className="text-xs font-medium text-purple-700">Chunk {idx + 1}</span>
                      <span className="text-xs font-medium px-2 py-1 bg-purple-100 text-purple-700 rounded">
                        Score: {(item.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-neutral-200 rounded h-1 mb-2">
                      <div
                        className="bg-primary h-1 rounded transition-all"
                        style={{ width: `${item.score * 100}%` }}
                      />
                    </div>
                    <p className="text-sm text-neutral-700 line-clamp-2">{String(item.chunk)}</p>
                  </div>
                ))}
              </div>
            </Card>
          ) : null}

          {/* Raw Response */}
          <Card className="bg-neutral-800">
            <h3 className="font-semibold text-white mb-3">Raw API Response</h3>
            <pre className="text-xs text-neutral-300 overflow-auto max-h-64 p-3 bg-neutral-900 rounded">
              {JSON.stringify(response as Record<string, unknown>, null, 2)}
            </pre>
          </Card>

          {/* Context */}
          {response && typeof response === 'object' && 'context' in response ? (
            <Card className="bg-green-50 border-green-200">
              <h3 className="font-semibold text-neutral-900 mb-3">Full Context</h3>
              <p className="text-sm text-neutral-700 whitespace-pre-wrap break-words">{String((response as Record<string, unknown>).context)}</p>
            </Card>
          ) : null}
        </div>
      )}
    </div>
  );
};
