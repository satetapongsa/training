import React from 'react';
import { Plus, Wifi, FolderPlus } from 'lucide-react';

export default function Topbar({ activeTab, wsConnected }) {
  const titles = {
    studio: 'หน้าหลัก: อัพโหลดโฟลเดอร์ & ตีกรอบวัตถุ',
    training: 'การเทรนโมเดล AI (Training)',
    inference: 'ทดสอบการตรวจจับ (Inference Testing)',
    models: 'ดาวน์โหลดโมเดล & ส่งออก ONNX',
  };

  return (
    <header className="topbar">
      <div className="topbar-title">
        <span>{titles[activeTab] || 'Vision Studio'}</span>
        {wsConnected && (
          <span className="badge badge-success" style={{ fontSize: '10px', marginLeft: '6px' }}>
            <Wifi size={12} /> พร้อมทำงาน
          </span>
        )}
      </div>
    </header>
  );
}
