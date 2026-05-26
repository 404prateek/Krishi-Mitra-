import { useState, useRef } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ── TEMPLATES ──────────────────────────────────────────────────────────────────
const TEMPLATES = {
  mausam: 'Kal aapke kshetra mein baarish ki sambhavna hai. Fasal ki suraksha ke liye upay karein aur seedha khule mein mat chhodein.',
  beej:   'Rabi fasal ke liye praamanik beej upalbdh hain. Nazdiki Krishi Kendra se sampark karein. Adhik jaankari ke liye 1800-XXX-XXXX.',
  mandi:  'Aaj ki mandi dar: Gehun ₹2,150/q | Dhan ₹1,940/q | Sarson ₹5,400/q. Apni fasal sahi samay par bechein.',
  yojana: 'PM Kisan Samman Nidhi ki agli kist jald aane wali hai. Apna aadhar aur bank link verify karein.',
  keeda:  'Aapke kshetra mein keede rog ki report mili hai. Anumoodit keetnashak ka upyog karein. Krishi adhikari se salah lein.',
};

// ── INITIAL MESSAGES ───────────────────────────────────────────────────────────
const INITIAL_MSGS = [
  { id: 1, phone: '9876543210', group: 'Wheat Zone',        preview: 'Kal baarish ki sambhavna hai, fasal ki suraksha karein…', lang: 'Hindi',   status: 'delivered', date: '26 May 2026', time: '10:32 AM' },
  { id: 2, phone: '9123456789', group: 'Rice Zone',         preview: 'Mandi mein gehun ka bhav aaj ₹2,150/quintal hai…',  lang: 'Punjabi', status: 'sent',      date: '25 May 2026', time: '3:15 PM'  },
  { id: 3, phone: '8800123456', group: 'All Farmers',       preview: 'PM Kisan Samman Nidhi ki kist aapke khate mein…',        lang: 'Hindi',   status: 'failed',    date: '24 May 2026', time: '9:00 AM'  },
  { id: 4, phone: '7700456123', group: 'Vegetable Growers', preview: 'Aphid aur thrips ka prabhaav badh raha hai, keetnashak…', lang: 'Hindi',   status: 'pending',   date: '26 May 2026', time: '11:00 AM' },
];

const STATUS_BADGE = {
  sent:      { bg: '#e8f5e9', color: '#2e7d32', label: 'Sent' },
  delivered: { bg: '#e3f2fd', color: '#1565c0', label: 'Delivered' },
  failed:    { bg: '#ffebee', color: '#c62828', label: 'Failed' },
  pending:   { bg: '#fff8e1', color: '#f57f17', label: 'Pending' },
};

const GROUPS = ['Sabhi Kisan (All Farmers)', 'Wheat Zone', 'Rice Zone', 'Vegetable Growers'];
const LANGS  = ['Hindi', 'English', 'Punjabi', 'Marathi', 'Telugu'];
const PAGE_SIZE = 4;

// Hardcoded group → number mapping (replace with dynamic DB lookup in production)
const GROUP_NUMBERS = {
  'Sabhi Kisan (All Farmers)': ['9876543210', '9123456789', '8800123456', '7700456123'],
  'Wheat Zone':                 ['9876543210', '9123456789'],
  'Rice Zone':                  ['8800123456'],
  'Vegetable Growers':          ['7700456123'],
};

// ── COMPONENT ──────────────────────────────────────────────────────────────────
export default function SMSAlertPage() {
  // Send form state
  const [recipients, setRecipients] = useState(['9876543210', '9123456789']);
  const [phoneInput, setPhoneInput] = useState('');
  const [group, setGroup]           = useState('');
  const [message, setMessage]       = useState('');
  const [lang, setLang]             = useState('Hindi');
  const [schedule, setSchedule]     = useState('');
  const [sendAlert, setSendAlert]   = useState(null);
  const [sending, setSending]       = useState(false);
  const phoneRef = useRef(null);

  // Stored messages state
  const [messages, setMessages]         = useState(INITIAL_MSGS);
  const [search, setSearch]             = useState('');
  const [filterStatus, setFilterStatus] = useState('');
  const [filterGroup, setFilterGroup]   = useState('');
  const [page, setPage]                 = useState(1);
  const [viewMsg, setViewMsg]           = useState(null);

  // Recipient tag helpers
  const addTag = () => {
    const val = phoneInput.trim();
    if (!val) return;
    if (!/^\d{10}$/.test(val)) { setSendAlert({ type: 'error', text: 'Valid 10-digit number dalein.' }); return; }
    if (!recipients.includes(val)) setRecipients(r => [...r, val]);
    setPhoneInput('');
    phoneRef.current?.focus();
  };
  const removeTag = (phone) => setRecipients(r => r.filter(p => p !== phone));

  // Send
  const sendSMS = async () => {
    if (!message.trim()) { setSendAlert({ type: 'error', text: 'Kripya sandesh likhein.' }); return; }
    if (message.length > 160) { setSendAlert({ type: 'error', text: 'Sandesh 160 characters se zyada nahi ho sakta.' }); return; }

    // Resolve final number list: explicit tags take priority, else expand group
    let targets = [...recipients];
    if (targets.length === 0 && group) {
      targets = GROUP_NUMBERS[group] || [];
    }
    if (targets.length === 0) { setSendAlert({ type: 'error', text: 'Recipient ya group chunein.' }); return; }

    const now = new Date();
    const dateStr = now.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
    const timeStr = now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
    const preview = message.length > 60 ? message.slice(0, 60) + '…' : message;

    // Optimistically add rows as "pending"
    const pendingRows = targets.map((phone, i) => ({
      id: Date.now() + i,
      phone,
      group: group || '—',
      preview,
      lang,
      status: 'pending',
      date: dateStr,
      time: timeStr,
      _fullMessage: message,
    }));
    setMessages(m => [...pendingRows, ...m]);
    setSending(true);
    setSendAlert(null);

    try {
      const res = await axios.post(`${API_BASE}/api/send-sms`, {
        numbers: targets,
        message,
        language: lang,
      });
      const { count, message_ids } = res.data;

      // Mark rows as "sent" and attach message_ids where available
      setMessages(m => m.map(row => {
        if (!pendingRows.find(pr => pr.id === row.id)) return row;
        const idx = pendingRows.findIndex(pr => pr.id === row.id);
        return { ...row, status: 'sent', messageId: message_ids?.[idx] || null };
      }));

      setSendAlert({ type: 'success', text: `✅ ${count} SMS successfully bhej diya gaya!` });
      setMessage(''); setRecipients([]); setGroup(''); setSchedule('');
    } catch (err) {
      const errText = err.response?.data?.detail || err.message || 'SMS send nahi hua. Dobara try karein.';

      // Mark rows as "failed"
      setMessages(m => m.map(row =>
        pendingRows.find(pr => pr.id === row.id) ? { ...row, status: 'failed' } : row
      ));
      setSendAlert({ type: 'error', text: `❌ ${errText}` });
    } finally {
      setSending(false);
      setTimeout(() => setSendAlert(null), 6000);
    }
  };

  const retry = async (id) => {
    const row = messages.find(m => m.id === id);
    if (!row) return;

    setMessages(m => m.map(x => x.id === id ? { ...x, status: 'pending' } : x));

    try {
      await axios.post(`${API_BASE}/api/send-sms`, {
        numbers:  [row.phone],
        message:  row._fullMessage || row.preview,
        language: row.lang,
      });
      setMessages(m => m.map(x => x.id === id ? { ...x, status: 'sent' } : x));
    } catch (err) {
      setMessages(m => m.map(x => x.id === id ? { ...x, status: 'failed' } : x));
    }
  };

  // Filtered + paginated
  const filtered = messages.filter(m => {
    const q = search.toLowerCase();
    const matchSearch = !q || m.phone.includes(q) || m.preview.toLowerCase().includes(q) || m.group.toLowerCase().includes(q);
    const matchStatus = !filterStatus || m.status === filterStatus;
    const matchGroup  = !filterGroup  || m.group === filterGroup;
    return matchSearch && matchStatus && matchGroup;
  });
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paged = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
  const charLen = message.length;

  return (
    <div style={S.page}>
      <div style={S.container}>

        {/* SEND SMS CARD */}
        <div style={S.card}>
          <div style={S.cardHeader}>
            <span>📤</span> SMS Bhejein (Send SMS)
          </div>
          <div style={S.cardBody}>

            {sendAlert && (
              <div style={{ ...S.alert, background: sendAlert.type === 'success' ? '#e8f5e9' : '#ffebee', color: sendAlert.type === 'success' ? '#1b5e20' : '#b71c1c', borderColor: sendAlert.type === 'success' ? '#a5d6a7' : '#ef9a9a' }}>
                {sendAlert.text}
              </div>
            )}

            {/* Recipients */}
            <div style={S.formGroup}>
              <label style={S.label}>Praaptakarta (Recipients)</label>
              <div style={S.recipRow}>
                <input
                  ref={phoneRef}
                  type="tel"
                  style={S.input}
                  value={phoneInput}
                  onChange={e => setPhoneInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); addTag(); } }}
                  placeholder="10-digit mobile number dalein, Enter dabayein…"
                  maxLength={10}
                />
                <button style={S.addBtn} onClick={addTag}>+ Add</button>
              </div>
              {recipients.length > 0 && (
                <div style={S.tagList}>
                  {recipients.map(p => (
                    <span key={p} style={S.tag}>
                      {p}
                      <button style={S.tagX} onClick={() => removeTag(p)}>✕</button>
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Group */}
            <div style={S.formGroup}>
              <label style={S.label}>Ya Group Chunein (Or select a group)</label>
              <select style={S.input} value={group} onChange={e => setGroup(e.target.value)}>
                <option value="">— Group chunein —</option>
                {GROUPS.map(g => <option key={g}>{g}</option>)}
              </select>
            </div>

            {/* Templates */}
            <div style={S.formGroup}>
              <label style={S.label}>Quick Templates</label>
              <div style={S.templateStrip}>
                {[
                  ['mausam', '🌦 Mausam Alert'],
                  ['beej',   '🌱 Beej Advisory'],
                  ['mandi',  '💰 Mandi Bhav'],
                  ['yojana', '📋 Sarkari Yojana'],
                  ['keeda',  '🐛 Keeda Rog Alert'],
                ].map(([key, lbl]) => (
                  <button key={key} style={S.tplBtn} onClick={() => setMessage(TEMPLATES[key])}>
                    {lbl}
                  </button>
                ))}
              </div>
            </div>

            {/* Message */}
            <div style={S.formGroup}>
              <label style={S.label}>Sandesh (Message)</label>
              <textarea
                style={{ ...S.input, minHeight: 100, resize: 'vertical' }}
                placeholder="Apna sandesh yahan likhein…"
                value={message}
                onChange={e => setMessage(e.target.value)}
              />
              <div style={{ ...S.charCount, ...(charLen > 140 ? S.charWarn : {}) }}>
                {charLen} / 160 characters
              </div>
            </div>

            {/* Language */}
            <div style={S.formGroup}>
              <label style={S.label}>Bhasha (Language)</label>
              <select style={S.input} value={lang} onChange={e => setLang(e.target.value)}>
                {LANGS.map(l => <option key={l}>{l}</option>)}
              </select>
            </div>

            {/* Schedule */}
            <div style={S.formGroup}>
              <label style={S.label}>Schedule (Optional)</label>
              <input
                type="datetime-local"
                style={S.input}
                value={schedule}
                onChange={e => setSchedule(e.target.value)}
              />
            </div>

            <button style={{ ...S.sendBtn, opacity: sending ? 0.7 : 1, cursor: sending ? 'not-allowed' : 'pointer' }} onClick={sendSMS} disabled={sending}>
              {sending ? '⏳ Bhej rahe hain…' : '📨 SMS Bhejein'}
            </button>

          </div>
        </div>

        {/* STORED MESSAGES CARD */}
        <div style={S.card}>
          <div style={S.cardHeader}>
            <span>📋</span> Bheje Gaye SMS (Sent / Stored Messages)
          </div>

          {/* Filters */}
          <div style={S.filters}>
            <input
              type="text"
              style={{ ...S.input, flex: 1, minWidth: 140 }}
              placeholder="🔍 Message ya number search karein…"
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(1); }}
            />
            <select style={{ ...S.input, flex: 1, minWidth: 130 }} value={filterStatus} onChange={e => { setFilterStatus(e.target.value); setPage(1); }}>
              <option value="">Sabhi Status</option>
              <option value="sent">Sent</option>
              <option value="delivered">Delivered</option>
              <option value="failed">Failed</option>
              <option value="pending">Pending</option>
            </select>
            <select style={{ ...S.input, flex: 1, minWidth: 140 }} value={filterGroup} onChange={e => { setFilterGroup(e.target.value); setPage(1); }}>
              <option value="">Sabhi Groups</option>
              {GROUPS.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>

          {/* Table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={S.table}>
              <thead>
                <tr>
                  {['#', 'Mobile', 'Group', 'Sandesh (Preview)', 'Bhasha', 'Status', 'Taarikh', 'Action'].map(h => (
                    <th key={h} style={S.th}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {paged.length === 0 ? (
                  <tr>
                    <td colSpan={8} style={{ textAlign: 'center', padding: '40px 20px', color: '#aaa' }}>
                      <span style={{ fontSize: 32, display: 'block', marginBottom: 8 }}>📭</span>
                      Koi message nahi mila
                    </td>
                  </tr>
                ) : paged.map((m, i) => {
                  const badge = STATUS_BADGE[m.status] || STATUS_BADGE.pending;
                  return (
                    <tr key={m.id} style={S.tr}>
                      <td style={S.td}>{(page - 1) * PAGE_SIZE + i + 1}</td>
                      <td style={S.td}>{m.phone}</td>
                      <td style={S.td}>{m.group}</td>
                      <td style={{ ...S.td, maxWidth: 260, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: '#555' }}>{m.preview}</td>
                      <td style={S.td}>{m.lang}</td>
                      <td style={S.td}>
                        <span style={{ ...S.badge, background: badge.bg, color: badge.color }}>{badge.label}</span>
                      </td>
                      <td style={S.td}>{m.date}<br /><small style={{ color: '#aaa' }}>{m.time}</small></td>
                      <td style={S.td}>
                        {m.status === 'failed'
                          ? <button style={S.actionBtn} onClick={() => retry(m.id)}>🔁 Retry</button>
                          : <button style={S.actionBtn} onClick={() => setViewMsg(m)}>👁 View</button>
                        }
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div style={S.pagination}>
            <span style={{ color: '#666', fontSize: 13 }}>
              {filtered.length === 0
                ? '0 messages'
                : `${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, filtered.length)} of ${filtered.length} messages`}
            </span>
            <div style={{ display: 'flex', gap: 6 }}>
              <button style={S.pageBtn} disabled={page === 1} onClick={() => setPage(p => p - 1)}>‹ Prev</button>
              {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1).map(p => (
                <button key={p} style={{ ...S.pageBtn, ...(p === page ? S.pageBtnActive : {}) }} onClick={() => setPage(p)}>{p}</button>
              ))}
              <button style={S.pageBtn} disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>Next ›</button>
            </div>
          </div>
        </div>

      </div>

      {/* View Message Modal */}
      {viewMsg && (
        <div style={S.modalBackdrop} onClick={() => setViewMsg(null)}>
          <div style={S.modal} onClick={e => e.stopPropagation()}>
            <div style={S.modalHeader}>
              <span style={{ fontWeight: 700, fontSize: 16, color: '#1b5e20' }}>📩 Message Details</span>
              <button style={S.modalClose} onClick={() => setViewMsg(null)}>✕</button>
            </div>
            <div style={S.modalBody}>
              {[['Mobile', viewMsg.phone], ['Group', viewMsg.group], ['Language', viewMsg.lang], ['Date', `${viewMsg.date} ${viewMsg.time}`]].map(([k, v]) => (
                <div key={k} style={S.modalRow}>
                  <span style={S.modalKey}>{k}:</span>
                  <span>{v}</span>
                </div>
              ))}
              <div style={S.modalRow}><span style={S.modalKey}>Status:</span><span style={{ color: STATUS_BADGE[viewMsg.status]?.color }}>{STATUS_BADGE[viewMsg.status]?.label}</span></div>
              <div style={{ ...S.modalRow, flexDirection: 'column', alignItems: 'flex-start', gap: 6 }}>
                <span style={S.modalKey}>Message:</span>
                <div style={{ background: '#f1f8e9', border: '1px solid #c8e6c9', borderRadius: 8, padding: '12px 14px', fontSize: 14, color: '#2d2d2d', lineHeight: 1.6, width: '100%' }}>
                  {viewMsg.preview}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── STYLES ─────────────────────────────────────────────────────────────────────
const S = {
  page:      { background: '#f0f4f0', minHeight: 'calc(100vh - 108px)', fontFamily: "'Segoe UI', sans-serif", color: '#2d2d2d' },
  container: { maxWidth: 900, margin: '0 auto', padding: '30px 16px 48px', display: 'flex', flexDirection: 'column', gap: 28 },

  card:       { background: '#fff', borderRadius: 12, boxShadow: '0 2px 10px rgba(0,0,0,0.08)', overflow: 'hidden' },
  cardHeader: { background: '#388e3c', color: '#fff', padding: '14px 20px', fontSize: 16, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 },
  cardBody:   { padding: '24px 20px' },

  alert:      { padding: '10px 14px', borderRadius: 8, border: '1px solid', marginBottom: 16, fontSize: 14, fontWeight: 500 },

  formGroup:  { marginBottom: 18 },
  label:      { display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.4px' },
  input:      { width: '100%', padding: '10px 14px', border: '1.5px solid #ddd', borderRadius: 8, fontSize: 14, color: '#333', background: '#fafafa', outline: 'none', boxSizing: 'border-box', fontFamily: 'inherit' },

  recipRow:   { display: 'flex', gap: 10 },
  addBtn:     { background: '#2e7d32', color: '#fff', border: 'none', padding: '10px 16px', borderRadius: 8, fontSize: 13, fontWeight: 600, cursor: 'pointer', whiteSpace: 'nowrap', flexShrink: 0 },

  tagList:    { display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 },
  tag:        { background: '#e8f5e9', color: '#2e7d32', border: '1px solid #a5d6a7', borderRadius: 20, padding: '3px 10px', fontSize: 13, display: 'flex', alignItems: 'center', gap: 5 },
  tagX:       { background: 'none', border: 'none', color: '#888', cursor: 'pointer', fontSize: 13, lineHeight: 1, padding: 0 },

  templateStrip: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  tplBtn:     { background: '#f1f8e9', border: '1px solid #aed581', color: '#558b2f', padding: '5px 12px', borderRadius: 20, fontSize: 12, cursor: 'pointer' },

  charCount:  { textAlign: 'right', fontSize: 12, color: '#888', marginTop: 4 },
  charWarn:   { color: '#e65100' },

  sendBtn:    { background: '#2e7d32', color: '#fff', border: 'none', padding: '12px 28px', borderRadius: 8, fontSize: 15, fontWeight: 600, cursor: 'pointer', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 },

  filters:    { display: 'flex', gap: 10, flexWrap: 'wrap', padding: '16px 20px', borderBottom: '1px solid #eee' },

  table:      { width: '100%', borderCollapse: 'collapse', fontSize: 13.5 },
  th:         { background: '#f5f5f5', padding: '11px 14px', textAlign: 'left', fontWeight: 600, color: '#444', borderBottom: '2px solid #e0e0e0', whiteSpace: 'nowrap' },
  tr:         { borderBottom: '1px solid #f0f0f0' },
  td:         { padding: '11px 14px', verticalAlign: 'middle' },

  badge:      { display: 'inline-block', padding: '3px 9px', borderRadius: 20, fontSize: 11.5, fontWeight: 600 },
  actionBtn:  { background: 'none', border: '1px solid #ddd', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', fontSize: 12, color: '#555' },

  pagination: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '14px 20px', borderTop: '1px solid #eee' },
  pageBtn:    { border: '1px solid #ddd', background: '#fff', borderRadius: 6, padding: '5px 11px', cursor: 'pointer', fontSize: 13 },
  pageBtnActive: { background: '#2e7d32', color: '#fff', borderColor: '#2e7d32' },

  modalBackdrop: { position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 },
  modal:      { background: '#fff', borderRadius: 14, width: '100%', maxWidth: 460, boxShadow: '0 12px 48px rgba(0,0,0,0.2)', overflow: 'hidden' },
  modalHeader:{ background: '#e8f5e9', padding: '16px 20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #c8e6c9' },
  modalClose: { background: 'none', border: 'none', fontSize: 16, cursor: 'pointer', color: '#555' },
  modalBody:  { padding: '20px', display: 'flex', flexDirection: 'column', gap: 12 },
  modalRow:   { display: 'flex', alignItems: 'center', gap: 10, fontSize: 14 },
  modalKey:   { fontWeight: 600, color: '#555', minWidth: 80, fontSize: 13 },
};
