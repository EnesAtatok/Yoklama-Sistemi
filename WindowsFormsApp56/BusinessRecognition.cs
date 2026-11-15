using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace WindowsFormsApp56
{
    /// <summary>
    /// Yüz bulma (Haar cascade) ve tanıma işini birleştiren sınıf.
    /// Form sadece buradaki Process metodunu çağırır.
    /// </summary>
    public class BusinessRecognition : IDisposable
    {
        private readonly CascadeClassifier _cascade;
        public Classifier_Train Classifier { get; private set; }

        public BusinessRecognition(string cascadePath, string facesFolder)
        {
            if (!File.Exists(cascadePath))
                throw new FileNotFoundException("haarcascade_frontalface_default.xml bulunamadı.", cascadePath);

            // OpenCV, Windows'ta ASCII dışı karakter içeren yolları (ör. "Masaüstü") açamaz.
            // Bu yüzden XML'i .NET ile okuyup bellekten yüklüyoruz.
            _cascade = new CascadeClassifier();
            string xml = File.ReadAllText(cascadePath);
            using (var fs = new FileStorage(xml, FileStorage.Mode.Read | FileStorage.Mode.Memory))
            {
                if (!_cascade.Read(fs.GetFirstTopLevelNode()))
                    throw new InvalidOperationException("Haar cascade dosyası yüklenemedi: " + cascadePath);
            }

            Classifier = new Classifier_Train(facesFolder);
            Classifier.Train(); // klasörde kayıtlı yüz varsa açılışta eğit
        }

        public Rectangle[] DetectFaces(Mat frame)
        {
            using (var gray = new Mat())
            {
                CvInvoke.CvtColor(frame, gray, ColorConversion.Bgr2Gray);
                CvInvoke.EqualizeHist(gray, gray);
                return _cascade.DetectMultiScale(gray, 1.2, 5, new Size(80, 80));
            }
        }

        /// <summary>
        /// Karedeki yüzleri bulur, çerçeve çizer; recognize true ise isim yazar.
        /// En büyük yüzü lastFace olarak döndürür (kayıt için).
        /// </summary>
        public int Process(Mat frame, bool recognize, out Mat lastFace)
        {
            lastFace = null;
            var faces = DetectFaces(frame);
            Rectangle biggest = Rectangle.Empty;
            using (var clean = frame.Clone()) // çerçeve çizilmemiş kopya
            {
            foreach (var rect in faces)
            {
                if (rect.Width * rect.Height > biggest.Width * biggest.Height) biggest = rect;

                string label = "Yuz";
                var color = new MCvScalar(0, 200, 0);

                if (recognize && Classifier.IsTrained)
                {
                    using (var face = new Mat(clean, rect))
                    {
                        double dist;
                        label = Classifier.Recognize(face, out dist);
                        if (label == Classifier_Train.Unknown) color = new MCvScalar(0, 0, 255);
                        else label += " (" + dist.ToString("0") + ")";
                    }
                }

                CvInvoke.Rectangle(frame, rect, color, 2);
                CvInvoke.PutText(frame, ToAscii(label),new Point(rect.X, Math.Max(rect.Y - 8, 15)),
                    FontFace.HersheySimplex, 0.8, color, 2);
            }

            if (!biggest.IsEmpty)
                using (var roi = new Mat(clean, biggest))
                    lastFace = roi.Clone();
            }

            return faces.Length;
        }

        /// <summary>OpenCV'nin Hershey fontu Türkçe harfleri çizemez ("?" çıkar); en yakın ASCII harfe çevir.</summary>
        private static string ToAscii(string text)
        {
            const string tr = "çÇğĞıİöÖşŞüÜ";
            const string en = "cCgGiIoOsSuU";
            var chars = text.ToCharArray();
            for (int i = 0; i < chars.Length; i++)
            {
                int k = tr.IndexOf(chars[i]);
                if (k >= 0) chars[i] = en[k];
                else if (chars[i] > 127) chars[i] = '?';
            }
            return new string(chars);
        }

        public void Dispose()
        {
            _cascade?.Dispose();
            Classifier?.Dispose();
        }
    }
}
