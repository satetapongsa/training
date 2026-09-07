import React from 'react';

export default function Topbar({ activeTab }) {
  const titles = {
    studio: 'หน้าหลัก: อัปโหลดไฟล์ ZIP รูปภาพวัตถุ & ตีกรอบเรียนรู้',
    training: 'การเทรนโมเดล AI (Training)',
    inference: 'ทดสอบการตรวจจับ (Inference Testing)',
  };

  return (
    <header className="topbar">
      <div className="topbar-title">
        <span>{titles[activeTab] || 'Vision Studio'}</span>
      </div>
    </header>
  );
}
