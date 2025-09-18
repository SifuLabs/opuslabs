# Video Editing Copilot - FFmpeg Installation & Setup Guide

## 🚨 Quick Fix: FFmpeg Not Found Error

The error `[WinError 2] The system cannot find the file specified` means FFmpeg is not installed on your system.

## 🔧 Solution 1: Install FFmpeg (Recommended)

### Windows - Easy Installation:
1. **Download FFmpeg**:
   - Go to: https://www.gyan.dev/ffmpeg/builds/
   - Download "release builds" → "ffmpeg-release-essentials.zip"
   
2. **Install FFmpeg**:
   ```bash
   # Extract to: C:\ffmpeg
   # Add to PATH: C:\ffmpeg\bin
   ```

3. **Add to PATH**:
   - Press Windows + R → type `sysdm.cpl`
   - Advanced → Environment Variables
   - System Variables → Path → Edit → New
   - Add: `C:\ffmpeg\bin`
   - Click OK, restart PowerShell

4. **Test Installation**:
   ```bash
   ffmpeg -version
   ```

### Alternative: Use Chocolatey (if you have it)
```bash
choco install ffmpeg
```

### Alternative: Use Scoop (if you have it)
```bash
scoop install ffmpeg
```

## 🚀 Solution 2: Quick Workaround (No FFmpeg needed)

Use the fallback video processing that works without FFmpeg:

```bash
# Run the demo version that doesn't need FFmpeg
python standalone_demo.py "Create clips from my video"

# Or use the production version in demo mode
python main_production.py "Make clips from my presentation"
```

## 🔧 Solution 3: Upgrade the Code

I can modify the video processor to handle videos without FFmpeg using Python libraries only.

---

## ✅ After FFmpeg is Installed

Once FFmpeg is properly installed, your video processing will work perfectly:

```bash
# Test with actual video
python main_production.py

# Then when prompted, provide your video file path
```

## 🎯 Quick Test After Setup

```bash
# This should show FFmpeg version info
ffmpeg -version

# This should work with your video
python main_production.py "Create 3 clips from my video"
```

---

**The most reliable solution is installing FFmpeg properly - it's essential for professional video processing!** 🎬