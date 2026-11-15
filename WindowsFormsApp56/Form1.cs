using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Emgu.CV;

namespace WindowsFormsApp56
{
    public class Form1 : Form
    {
        private readonly PictureBox pictureBox = new PictureBox();
        private readonly TextBox txtName = new TextBox();
        private readonly Button btnCamera = new Button();
        private readonly Button btnSave = new Button();
        private readonly Button btnTrain = new Button();
        private readonly CheckBox chkRecognize = new CheckBox();
        private readonly Label lblStatus = new Label();
        private readonly Timer timer = new Timer();

        private VideoCapture _capture;
        private BusinessRecognition _recognition;
        private Mat _lastFace;

        public Form1()
        {
            BuildUi();
            Load += Form1_Load;
            FormClosing += Form1_FormClosing;
            timer.Interval = 40;
            timer.Tick += Timer_Tick;
        }

        private void BuildUi()
        {
            Text = "Yüz Tanıma";
            ClientSize = new Size(900, 520);
            StartPosition = FormStartPosition.CenterScreen;

            pictureBox.Location = new Point(12, 12);
            pictureBox.Size = new Size(640, 480);
            pictureBox.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBox.BackColor = Color.Black;

            var lblName = new Label { Text = "Kişi adı:", Location = new Point(670, 14), AutoSize = true };
            txtName.Location = new Point(670, 36);
            txtName.Width = 210;

            btnCamera.Text = "Kamerayı Başlat";
            btnCamera.Location = new Point(670, 76);
            btnCamera.Size = new Size(210, 34);
            btnCamera.Click += BtnCamera_Click;

            btnSave.Text = "Yüzü Kaydet";
            btnSave.Location = new Point(670, 120);
            btnSave.Size = new Size(210, 34);
            btnSave.Click += BtnSave_Click;

            btnTrain.Text = "Eğit";
            btnTrain.Location = new Point(670, 164);
            btnTrain.Size = new Size(210, 34);
            btnTrain.Click += BtnTrain_Click;

            chkRecognize.Text = "Tanımayı aç";
            chkRecognize.Location = new Point(670, 212);
            chkRecognize.AutoSize = true;

            lblStatus.Location = new Point(670, 250);
            lblStatus.Size = new Size(210, 240);

            Controls.AddRange(new Control[]
            {
                pictureBox, lblName, txtName, btnCamera, btnSave, btnTrain, chkRecognize, lblStatus
            });
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                _recognition = new BusinessRecognition(
                    Path.Combine(baseDir, "haarcascade_frontalface_default.xml"),
                    Path.Combine(baseDir, "TrainedFaces"));
                SetStatus("Hazır. Kayıtlı yüz: " + _recognition.Classifier.FaceCount);
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Başlatma hatası", MessageBoxButtons.OK, MessageBoxIcon.Error);
                btnCamera.Enabled = btnSave.Enabled = btnTrain.Enabled = false;
            }
        }

        private void BtnCamera_Click(object sender, EventArgs e)
        {
            if (timer.Enabled) { StopCamera(); return; }

            Cursor = Cursors.WaitCursor;
            try
            {
                _capture = OpenCamera(VideoCapture.API.DShow) ?? OpenCamera(VideoCapture.API.Any);
            }
            finally
            {
                Cursor = Cursors.Default;
            }
            if (_capture == null)
            {
                MessageBox.Show("Kamera açılamadı. Başka bir uygulama kamerayı kullanıyor olabilir.",
                    "Hata", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            timer.Start();
            btnCamera.Text = "Kamerayı Durdur";
        }

        private static VideoCapture OpenCamera(VideoCapture.API api)
        {
            try
            {
                var cap = new VideoCapture(0, api);
                if (cap.IsOpened) return cap;
                cap.Dispose();
            }
            catch (Exception) { }
            return null;
        }

        private void StopCamera()
        {
            timer.Stop();
            _capture?.Dispose();
            _capture = null;
            btnCamera.Text = "Kamerayı Başlat";
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            if (_capture == null || _recognition == null) return;
            try
            {
                using (var frame = _capture.QueryFrame())
                {
                    if (frame == null || frame.IsEmpty) return;

                    Mat face;
                    int count = _recognition.Process(frame, chkRecognize.Checked, out face);

                    _lastFace?.Dispose();
                    _lastFace = face;

                    var old = pictureBox.Image;
                    pictureBox.Image = frame.ToBitmap();
                    old?.Dispose();

                    lblStatus.Text = "Bulunan yüz: " + count +
                                     "\nKayıtlı yüz: " + _recognition.Classifier.FaceCount;
                }
            }
            catch (Exception ex)
            {
                StopCamera();
                MessageBox.Show("Görüntü işlenirken hata: " + ex.Message, "Hata",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void BtnSave_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(txtName.Text))
            {
                MessageBox.Show("Önce kişi adını yaz.");
                return;
            }
            if (_lastFace == null || _lastFace.IsEmpty)
            {
                MessageBox.Show("Kamerada yüz görünmüyor. Kamerayı başlat ve kameraya bak.");
                return;
            }
            try
            {
                _recognition.Classifier.SaveFace(_lastFace, txtName.Text);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Yüz kaydedilemedi: " + ex.Message, "Hata", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            SetStatus("Yüz kaydedildi. Bir kişi için 10-20 farklı poz kaydet, sonra 'Eğit'e bas.");
        }

        private void BtnTrain_Click(object sender, EventArgs e)
        {
            try
            {
                if (_recognition.Classifier.Train())
                {
                    SetStatus("Eğitim tamam. Resim sayısı: " + _recognition.Classifier.FaceCount +
                              "\nKişiler: " + string.Join(", ", _recognition.Classifier.Names));
                    chkRecognize.Checked = true;
                }
                else
                    SetStatus("Eğitilecek yüz resmi yok. Önce yüz kaydet.");
            }
            catch (Exception ex)
            {
                MessageBox.Show("Eğitim hatası: " + ex.Message, "Hata", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void SetStatus(string text)
        {
            lblStatus.Text = text;
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            StopCamera();
            _lastFace?.Dispose();
            _recognition?.Dispose();
        }
    }
}
