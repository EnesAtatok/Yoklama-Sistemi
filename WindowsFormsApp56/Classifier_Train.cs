using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Util;

namespace WindowsFormsApp56
{
    /// <summary>
    /// Kayıtlı yüz resimlerini (TrainedFaces klasörü) okuyup LBPH ile eğitir
    /// ve yeni yüzleri tanır. Dosya adı biçimi: Isim_1.png, Isim_2.png ...
    /// </summary>
    public class Classifier_Train : IDisposable
    {
        public static readonly Size FaceSize = new Size(100, 100);
        public const string Unknown = "Bilinmiyor";

        private LBPHFaceRecognizer _recognizer;
        private readonly List<string> _names = new List<string>();

        /// <summary>Bu mesafeden büyükse "Bilinmiyor" döner (küçük = daha benzer).</summary>
        public double Threshold { get; set; } = 80;

        public string FacesFolder { get; private set; }
        public bool IsTrained { get; private set; }
        public int FaceCount { get; private set; }
        public IReadOnlyList<string> Names { get { return _names; } }

        public Classifier_Train(string facesFolder)
        {
            FacesFolder = facesFolder;
            Directory.CreateDirectory(FacesFolder);
        }

        /// <summary>Yüzü 100x100 gri olarak hazırlar (kayıt ve tanıma için aynı işlem).</summary>
        public static Mat Prepare(Mat faceBgr)
        {
            var gray = new Mat();
            if (faceBgr.NumberOfChannels == 1)
                faceBgr.CopyTo(gray);
            else
                CvInvoke.CvtColor(faceBgr, gray, ColorConversion.Bgr2Gray);

            CvInvoke.Resize(gray, gray, FaceSize, 0, 0, Inter.Cubic);
            CvInvoke.EqualizeHist(gray, gray);
            return gray;
        }

        /// <summary>Yeni bir yüz resmini klasöre kaydeder.</summary>
        public string SaveFace(Mat faceBgr, string name)
        {
            name = Sanitize(name);
            if (name.Length == 0) throw new ArgumentException("Geçerli bir isim girin.", nameof(name));

            // Silinmiş dosyalar varsa üzerine yazmamak için ilk boş numarayı bul
            int next = 1;
            string path;
            while (File.Exists(path = Path.Combine(FacesFolder, name + "_" + next + ".png"))) next++;

            // CvInvoke.Imwrite ASCII dışı yollarda başarısız olur; belleğe kodlayıp .NET ile yaz
            using (var prepared = Prepare(faceBgr))
            using (var buf = new VectorOfByte())
            {
                CvInvoke.Imencode(".png", prepared, buf);
                File.WriteAllBytes(path, buf.ToArray());
            }
            return path;
        }

        /// <summary>Klasördeki tüm resimlerle modeli eğitir. Resim yoksa false döner.</summary>
        public bool Train()
        {
            _names.Clear();
            IsTrained = false;
            FaceCount = 0;

            var files = Directory.GetFiles(FacesFolder, "*.png").OrderBy(f => f).ToArray();
            if (files.Length == 0) return false;

            var mats = new List<Mat>();
            var labels = new List<int>();
            try
            {
                foreach (var file in files)
                {
                    var img = ReadGray(file);
                    if (img == null) continue;
                    CvInvoke.Resize(img, img, FaceSize);

                    string baseName = Path.GetFileNameWithoutExtension(file);
                    int idx = baseName.LastIndexOf('_');
                    string person = idx > 0 ? baseName.Substring(0, idx) : baseName;

                    int label = _names.IndexOf(person);
                    if (label < 0) { _names.Add(person); label = _names.Count - 1; }

                    mats.Add(img);
                    labels.Add(label);
                }

                if (labels.Count == 0) return false;

                _recognizer?.Dispose();
                _recognizer = new LBPHFaceRecognizer(1, 8, 8, 8, double.MaxValue);
                using (var images = new VectorOfMat(mats.ToArray()))
                using (var labelVec = new VectorOfInt(labels.ToArray()))
                    _recognizer.Train(images, labelVec);
            }
            finally
            {
                foreach (var m in mats) m.Dispose();
            }

            FaceCount = labels.Count;
            IsTrained = true;
            return true;
        }

        /// <summary>Verilen yüzü tanır. Eşleşme yoksa "Bilinmiyor" döner.</summary>
        public string Recognize(Mat faceBgr, out double distance)
        {
            distance = double.NaN;
            if (!IsTrained) return Unknown;

            using (var prepared = Prepare(faceBgr))
            {
                var result = _recognizer.Predict(prepared);
                distance = result.Distance;
                if (result.Label >= 0 && result.Label < _names.Count && result.Distance <= Threshold)
                    return _names[result.Label];
                return Unknown;
            }
        }

        /// <summary>Resmi gri tonlamalı okur (CvInvoke.Imread ASCII dışı yollarda çalışmaz).</summary>
        private static Mat ReadGray(string file)
        {
            try
            {
                var img = new Mat();
                CvInvoke.Imdecode(File.ReadAllBytes(file), ImreadModes.Grayscale, img);
                if (img.IsEmpty) { img.Dispose(); return null; }
                return img;
            }
            catch (Exception)
            {
                return null; // bozuk/okunamayan dosyayı atla
            }
        }

        private static string Sanitize(string name)
        {
            name = (name ?? "").Trim();
            foreach (char c in Path.GetInvalidFileNameChars()) name = name.Replace(c, '-');
            return name.Replace('_', '-');
        }

        public void Dispose()
        {
            _recognizer?.Dispose();
        }
    }
}
