# video_insertion

```bash
video_insertion/
├── modules/                       # 核心邏輯與模組（每個步驟為子目錄）
│   ├── detection/                 # Step 1: 物件偵測
│   │
│   ├── occlusion/                # Step 2: 遮擋處理
│   │
│   ├── matching/                 # Step 3: 語意與深度匹配
│   │   ├── semantic.py
│   │   ├── depth.py
│   │   └── utils.py
│   │
│   ├── validation/               # Step 4: VLM 驗證
│   │   └──/vlm                    
│   │
│   ├── resizing/                 # Step 5: 尺寸估計與插入
│   │   ├── size_estimator.py
│   │   └── paste.py
│   │
│   ├── models/
│   │   ├── detection_model.py
│   │   ├── matching_model.py
│   │   ├── validation_model.py
│   │   ├── resizing_model.py
│   │   └── occlusion_model.py
│   │
│   └── configs/
│
├── data/                          # 所有資料和metadata
│   ├── objects/
│   ├── videos/                    # 擷取影片的第一幀
│   └── masks/                     # 需要存照片的 mask 的時候使用
│   
├── utils/                         # 非核心工具（如視覺化、數據轉換）
│   └── visualization/
│
│
├── scripts/                   # 可執行的pipeline腳本
│   ├── run_detection.py
│   ├── run_occlusion.py
│   └── run_full_insertion.py
│
├── tests/                         # 單元測試
├── docs/                          # 說明文件
├── main.py                        # 主執行程式
├── README.md
└── requirements.txt
```

