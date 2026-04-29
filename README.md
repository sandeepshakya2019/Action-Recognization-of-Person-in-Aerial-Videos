# 🚁 Aerial Multi-Label Action Recognition Pipeline

## 📌 Overview
This project presents a **person-centric multi-label action recognition system** for aerial videos captured by drones. It combines **object detection, tracking, and temporal modeling** to recognize multiple actions performed by individuals in complex real-world environments.

The system is designed to handle:
- Small object sizes in aerial imagery  
- Occlusions and identity switching  
- Camera motion and dynamic backgrounds  
- Multi-label action classification  

---

## 🧠 Pipeline Architecture

![Pipeline]([https://via.placeholder.com/800x400.png?text=Pipeline+Architecture](https://github.com/sandeepshakya2019/Action-Recognization-of-Person-in-Aerial-Videos/blob/main/images/pipeline-diagram.png))

### 🔁 Workflow
1. **Detection** → YOLOv8 detects persons  
2. **Tracking** → DeepSORT + Particle Filter assigns IDs  
3. **Sequence Formation** → Person-wise frame clips  
4. **Action Recognition** → TimeSformer predicts actions  

---

## ⚙️ Key Components

### 🔍 Object Detection
- Model: YOLOv8  
- Output: Bounding boxes + confidence scores  

### 🎯 Tracking Module
- DeepSORT (appearance + motion)
- Particle Filter (robust tracking)

### 🎬 Temporal Modeling
- Model: TimeSformer  
- Input: 16-frame sequences  
- Output: Multi-label action predictions  

---

## 📊 Results

### 📈 Detection Performance
- mAP@0.5: **0.83**  
- Precision: **0.92**  

### 🎯 Action Recognition Performance
- Strong performance on frequent classes  
- Micro-F1 > Macro-F1  

---

## 📸 Results Visualization

### 🔹 Detection + Tracking Output
<img width="276" height="129" alt="image" src="https://github.com/user-attachments/assets/8b36d2be-3b8f-4c0c-8b15-d76b7967b28a" />


### 🔹 Action Recognition Output
<img width="293" height="197" alt="image" src="https://github.com/user-attachments/assets/4af6769c-a965-4288-9ec2-b01f1d4dbd81" />


---

## 📄 Reports & Presentation

- 📑 **Final Report:**  
  https://github.com/sandeepshakya2019/Action-Recognization-of-Person-in-Aerial-Videos/blob/main/reports-presentation/phase2/Sandeep_Kumar_MTP_Final_Report.pdf  

- 🎥 **Final Presentation:**  
  https://github.com/sandeepshakya2019/Action-Recognization-of-Person-in-Aerial-Videos/blob/main/reports-presentation/phase2/Sandeep_Kumar_MTP_Final_Presentation.pdf  

---

## 🧪 Challenges Addressed

- Small-scale object detection  
- Identity consistency during occlusion  
- Temporal action modeling  
- Multi-label classification  

---
