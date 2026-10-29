import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/Sidebar';
import { TopBar } from '../components/TopBar';
import { 
  History, 
  Search, 
  Filter, 
  ArrowRight, 
  FileText, 
  CheckCircle, 
  XCircle, 
  RefreshCw
} from 'lucide-react';
import { getDeepfakeHistory } from '../services/api';

export function HistoryPage() {
  const navigate = useNavigate();

  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('ALL'); // ALL, REAL, FAKE
  const [currentPage, setCurrentPage] = useState(1);

  const itemsPerPage = 10;

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const res = await getDeepfakeHistory();
      setRecords(res?.history || []);
    } catch (err) {
      console.error("Failed to load history:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  // Filter & Search Logic
  const filteredRecords = records.filter(rec => {
    const payload = rec?.payload || {};
    const det = payload?.detection_result || payload?.result || rec?.result || {};
    const meta = payload?.input_meta || payload?.input || rec?.input || {};
    const analysisId = payload?.analysis_id || rec?.analysis_id || '';
    const filename = meta?.filename || '';

    const matchesSearch = 
      analysisId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      filename.toLowerCase().includes(searchTerm.toLowerCase());

    const classification = det?.classification || det?.label || '';
    const isFake = classification === 'DEEPFAKE' || classification === 'FAKE';

    if (filterType === 'REAL') return matchesSearch && !isFake;
    if (filterType === 'FAKE') return matchesSearch && isFake;
    return matchesSearch;
  });

  const totalPages = Math.ceil(filteredRecords.length / itemsPerPage) || 1;
  const paginatedRecords = filteredRecords.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  return (
    <div className="flex min-h-screen bg-[#07090e] text-gray-200 font-sans">
      <Sidebar />

      <div className="flex-1 flex flex-col min-w-0">
        <TopBar title="Analysis History" subtitle="Enterprise audit log of completed DeepQShield security inspections" />

        <main className="flex-1 p-8 space-y-8 max-w-7xl mx-auto w-full">
          {/* Header Controls */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl font-bold font-mono text-white flex items-center gap-2">
                <History className="w-6 h-6 text-blue-400" />
                <span>ANALYSIS HISTORY</span>
              </h2>
              <p className="text-xs text-gray-400 mt-1">
                Real-time security log of all evaluated image and video media records.
              </p>
            </div>

            <button
              onClick={fetchHistory}
              className="px-4 py-2 rounded-xl bg-[#0c1017] hover:bg-gray-800/80 border border-[#1f293d] text-xs font-mono text-gray-300 hover:text-white transition-all flex items-center gap-2 shadow-sm"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-400' : ''}`} />
              <span>Refresh History</span>
            </button>
          </div>

          {/* Search & Filter Bar */}
          <div className="p-4 rounded-2xl bg-[#0c1017] border border-[#1f293d] flex flex-col sm:flex-row items-center justify-between gap-4 font-mono text-xs shadow-2xl">
            {/* Search Input */}
            <div className="relative w-full sm:w-80">
              <Search className="w-4 h-4 text-gray-400 absolute left-3.5 top-3" />
              <input
                type="text"
                placeholder="Search Analysis ID or Filename..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                className="w-full bg-[#07090e] border border-[#1f293d] rounded-xl pl-10 pr-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
            </div>

            {/* Filter Buttons */}
            <div className="flex items-center gap-2 w-full sm:w-auto">
              <span className="text-gray-400 mr-2 flex items-center gap-1 font-medium">
                <Filter className="w-3.5 h-3.5 text-gray-400" /> Filter:
              </span>
              {['ALL', 'REAL', 'FAKE'].map(type => (
                <button
                  key={type}
                  onClick={() => { setFilterType(type); setCurrentPage(1); }}
                  className={`px-3 py-1.5 rounded-lg transition-all font-medium ${
                    filterType === type
                      ? 'bg-blue-600 text-white shadow-md font-bold'
                      : 'bg-[#07090e] text-gray-400 border border-[#1f293d] hover:bg-gray-800/80 hover:text-white'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* History Table */}
          <div className="rounded-2xl bg-[#0c1017] border border-[#1f293d] overflow-hidden shadow-2xl">
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-[#07090e] text-gray-400 border-b border-[#1f293d] uppercase text-[10px] tracking-wider">
                  <tr>
                    <th className="py-3.5 px-6">Analysis ID</th>
                    <th className="py-3.5 px-6">Filename</th>
                    <th className="py-3.5 px-6">Media Type</th>
                    <th className="py-3.5 px-6">Result</th>
                    <th className="py-3.5 px-6">Confidence</th>
                    <th className="py-3.5 px-6">Timestamp</th>
                    <th className="py-3.5 px-6">Status</th>
                    <th className="py-3.5 px-6 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800/80">
                  {loading && (
                    <tr>
                      <td colSpan="8" className="py-8 text-center text-gray-400">
                        Loading analysis records from DeepQShield database…
                      </td>
                    </tr>
                  )}

                  {!loading && paginatedRecords.length === 0 && (
                    <tr>
                      <td colSpan="8" className="py-12 text-center space-y-3">
                        <FileText className="w-10 h-10 text-gray-500 mx-auto" />
                        <p className="text-white font-bold">No Analysis History Records Found</p>
                        <p className="text-gray-400 text-[11px]">Perform authentication on `/auth` to generate inspection records.</p>
                      </td>
                    </tr>
                  )}

                  {!loading && paginatedRecords.map((rec, idx) => {
                    const payload = rec?.payload || {};
                    const det = payload?.detection_result || payload?.result || rec?.result || {};
                    const meta = payload?.input_meta || payload?.input || rec?.input || {};
                    const analysisId = payload?.analysis_id || rec?.analysis_id || 'N/A';
                    const filename = meta?.filename || 'media_frame.jpg';
                    const mediaType = meta?.media_kind || meta?.type || 'Image/Video';
                    const classification = det?.classification || det?.label || '';
                    const isFake = classification === 'DEEPFAKE' || classification === 'FAKE';
                    const confidence = det?.confidence ?? 0.0;
                    const timestamp = payload?.timestamp || new Date().toISOString();

                    return (
                      <tr key={idx} className="hover:bg-gray-800/40 transition-colors">
                        <td className="py-4 px-6 text-blue-400 font-bold truncate max-w-[140px]">{analysisId}</td>
                        <td className="py-4 px-6 text-gray-200 font-medium truncate max-w-[180px]">{filename}</td>
                        <td className="py-4 px-6 text-gray-400 uppercase text-[10px]">{mediaType}</td>
                        <td className="py-4 px-6">
                          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold ${
                            isFake
                              ? 'bg-rose-500/10 border border-rose-500/30 text-rose-400'
                              : 'bg-emerald-500/10 border border-emerald-500/30 text-emerald-400'
                          }`}>
                            {isFake ? <XCircle className="w-3 h-3" /> : <CheckCircle className="w-3 h-3" />}
                            {isFake ? 'MANIPULATED' : 'AUTHENTIC'}
                          </span>
                        </td>
                        <td className="py-4 px-6 text-white font-bold">{(confidence * 100).toFixed(1)}%</td>
                        <td className="py-4 px-6 text-gray-400 text-[11px]">{new Date(timestamp).toLocaleDateString()}</td>
                        <td className="py-4 px-6">
                          <span className="text-emerald-400 font-bold text-[10px]">VERIFIED</span>
                        </td>
                        <td className="py-4 px-6 text-right">
                          <button
                            onClick={() => navigate(`/analysis/${analysisId}`)}
                            className="text-blue-400 hover:text-blue-300 font-bold text-[11px] inline-flex items-center gap-1"
                          >
                            <span>Inspect</span>
                            <ArrowRight className="w-3 h-3" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Pagination Footer */}
            {!loading && filteredRecords.length > 0 && (
              <div className="p-4 bg-[#07090e] border-t border-[#1f293d] flex items-center justify-between text-xs font-mono text-gray-400">
                <span>Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(filteredRecords.length, currentPage * itemsPerPage)} of {filteredRecords.length} records</span>
                <div className="flex items-center gap-2">
                  <button
                    disabled={currentPage === 1}
                    onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                    className="px-3 py-1 rounded-lg bg-[#0c1017] border border-[#1f293d] disabled:opacity-40 hover:bg-gray-800/80 text-gray-300 font-medium"
                  >
                    Previous
                  </button>
                  <span>Page {currentPage} of {totalPages}</span>
                  <button
                    disabled={currentPage === totalPages}
                    onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                    className="px-3 py-1 rounded-lg bg-[#0c1017] border border-[#1f293d] disabled:opacity-40 hover:bg-gray-800/80 text-gray-300 font-medium"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default HistoryPage;
