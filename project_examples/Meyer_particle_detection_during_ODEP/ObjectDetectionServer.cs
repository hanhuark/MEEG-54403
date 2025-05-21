using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using Newtonsoft.Json;

namespace SecondImageCreationByPixel
{
    public partial class ObjectDetectionServer : Form
    {

        private static int distanceToMaintain=50;

        public bool StopServer;

        private frmMain MainForm;

        JsonSerializer jsonSerializer= new JsonSerializer();

        public bool ServerListening=false;

        private TcpListener ObjectDetectionServerObject;

        List<ShapeProperties> AutomatedLightSettingsDefaultValues = new List<ShapeProperties>();

        static List<CaptureLocation> Class1CaptureLocations = new List<CaptureLocation>();

        static List<CaptureLocation> Class2CaptureLocations = new List<CaptureLocation>();

        List<DetectedObject> detectedObjects1 = new List<DetectedObject>();

        private int portNumber=999;
        private int PortNumber { set {  portNumber = value; UpdateIPAddressShowOutput(); } get{ return portNumber; } }

        private IPAddress iPAddress=IPAddress.Loopback;
        private IPAddress IPAddress { set {  iPAddress = value; UpdateIPAddressShowOutput(); } get { return iPAddress; } }

        private void UpdateIPAddressShowOutput()
        {
            lbIPAddressDisplay.Text = "Complete IP Address:"+IPAddress.ToString() + ":" + PortNumber.ToString();
        }

        public ObjectDetectionServer(frmMain MainForm)
        {
            InitializeComponent();
            UpdateIPAddressShowOutput();
            this.FormClosing += ObjectDetectionServer_FormClosing;
            this.MainForm = MainForm;
            AutomatedLightSettingsDefaultValues.Add(Capture_Properties_Form.GetShapePropertiesSettings(Properties.DefaultLightSettings.Default));
            AutomatedLightSettingsDefaultValues.Add(Capture_Properties_Form.GetShapePropertiesSettings(Properties.DefaultLightSettings.Default));
            tbLRX.TypeOfInput = InputTextBox.InputTypeEnum.Lower_RightX;
            tbLRY.TypeOfInput = InputTextBox.InputTypeEnum.Lower_RightY;
            tbULX.TypeOfInput = InputTextBox.InputTypeEnum.Upper_LeftX;
            tbULY.TypeOfInput = InputTextBox.InputTypeEnum.Upper_LeftY;
            tbLRX.Text = Properties.AutomationSettings.Default.LowerRightCorner.X.ToString();
            tbLRY.Text = Properties.AutomationSettings.Default.LowerRightCorner.Y.ToString();
            tbULX.Text = Properties.AutomationSettings.Default.UpperLeftCorner.X.ToString();
            tbULY.Text = Properties.AutomationSettings.Default.UpperLeftCorner.Y.ToString();
        }

        private void ObjectDetectionServer_FormClosing(object sender, FormClosingEventArgs e)
        {
             e.Cancel = true;
        }

        private void radioButton2_CheckedChanged(object sender, EventArgs e)
        {
            IPAddress = IPAddress.Loopback;
        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            
            IPAddress = IPAddress.Parse(GetLocalIPAddress());
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            if(int.TryParse(tbPortNumber.Text,out int NewPort))
            {
                if(NewPort <= 65535 && NewPort > 0)
                {
                    PortNumber = NewPort;

                }
            }
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        public static string GetLocalIPAddress()
        {
            var host = Dns.GetHostEntry(Dns.GetHostName());
            foreach (var ip in host.AddressList)
            {
                if (ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                {
                    return ip.ToString();
                }
            }
            throw new Exception("No network adapters with an IPv4 address in the system!");
        }

        private void tbPortNumber_KeyPress(object sender, KeyPressEventArgs e)
        {
            if (!char.IsControl(e.KeyChar) && !char.IsDigit(e.KeyChar) && (e.KeyChar != '.'))
            {
                e.Handled = true;
            }


            if ((e.KeyChar == '.') && ((sender as TextBox).Text.IndexOf('.') > -1))
            {
                e.Handled = true;
            }
        }

        System.Timers.Timer timer = new System.Timers.Timer();
        private void DisconnectClientAndStartListening()
        {
            timer.Enabled = true;
            ServerListening = true;
            UpdateDataReadOutTextbox("Client Disconnected");
        }

        private void BeginListening()
        {
            timer.Elapsed += CheckForClientConnectionRequest;
            timer.Interval = 500;
            timer.Enabled = true;
            ServerListening = true;
            ObjectDetectionServerObject = new TcpListener(IPAddress, portNumber);
            try
            {
                ObjectDetectionServerObject.Start();
                //MessageBox.Show("Server Started");
            }
            catch(Exception e)
            {
                MessageBox.Show(e.Message);
            }
           
        }
        private void CheckForClientConnectionRequest(object sender,EventArgs args)
        {
            if (ObjectDetectionServerObject == null||!(ObjectDetectionServerObject.Server.IsBound))
            {
                timer.Enabled = false;
                return;
            }
            
            
            if (ObjectDetectionServerObject.Pending())
            {
                timer.Enabled = false;
                using (var connection = ObjectDetectionServerObject.AcceptTcpClient())
                {
                    using(var stream = connection.GetStream())
                    {
                        UpdateDataReadOutTextbox("New Client Connected");
                        stream.ReadTimeout = 10000;
                        byte[] ConnectionTestByte = new byte[256];

                        while (!StopServer && connection.Connected)
                        {
                            if (StopServer)
                            {
                                DisconnectClientAndStartListening();
                                return;
                            }
                            try
                            {
                                stream.WriteByte(0);
                            }
                            catch(System.IO.IOException e)
                            {
                                if(e.InnerException!=null&&e.InnerException.Message.Contains("An established connection was aborted by the software in your host machine"))
                                {
                                    StopServer = true;
                                }
                                Console.WriteLine(e.Message)
                            }
                            var datasize = connection.Available;
                            Byte[] bytes = new Byte[datasize];
                            try
                            {
                                var data = stream.Read(bytes, 0, bytes.Length);
                                if (data > 0)
                                {
                                    try
                                    {
                                        var jsonData = Encoding.UTF8.GetString(bytes);
                                        var detectedObjects=JsonConvert.DeserializeObject<DetectedObject>(jsonData);
                                        detectedObjects1.Add(detectedObjects);
                                        detectedObjects.SetID();
                                        CaptureAndMove(detectedObjects);
                                        UpdateDataReadOutTextbox(jsonData);
                                        Console.WriteLine("Incoming Data: " + jsonData);
                                    }
                                    catch (Exception e)
                                    {
                                        Console.WriteLine(e.Message)
                                    }

                                }
                            }
                            catch(System.IO.IOException e)
                            {
                                Console.WriteLine(e.Message)
                            }
                            catch(Exception ee)
                            {
                                Console.WriteLine(e.Message)
                            }
                        }
                        DisconnectClientAndStartListening();
                    }
                } 
            }
        }

        private void CaptureAndMove(DetectedObject detectedObject)
        {
            var upleft = Properties.AutomationSettings.Default.UpperLeftCorner;
            var downrigh = Properties.AutomationSettings.Default.LowerRightCorner;
            var xspan = Math.Abs(downrigh.X - upleft.X);
            var yspan = Math.Abs(downrigh.Y - upleft.Y);
            detectedObject.X = (int)Math.Round((double)(((double)detectedObject.X / (double)detectedObject.Xlimit)* (double)xspan))+ Properties.AutomationSettings.Default.UpperLeftCorner.X;
            detectedObject.Y = (int)Math.Round(((((double)detectedObject.Y / (double)detectedObject.Ylimit))* (double)yspan))+ Properties.AutomationSettings.Default.UpperLeftCorner.Y;
            detectedObject.Width = (int)Math.Round((((double)detectedObject.Width / (double)detectedObject.Xlimit)* (double)xspan));
            detectedObject.Height = (int)Math.Round((((double)detectedObject.Height / (double)detectedObject.Ylimit))* (double)yspan);

          
            if (cbEnableCapture.Checked && !checkIfRingExists(detectedObject.X, detectedObject.Y, detectedObject.Width, detectedObject.Height,detectedObject.Type))
            {
                if (detectedObject.Type == "6um" && !ObjectIsInObjectiveLocation(Objective1CaptureLocation, detectedObject))
                {
                    CreateCaptureLocation(detectedObject, Class1CaptureLocations, Objective1CaptureLocation,AutomatedLightSettingsDefaultValues[0]);                    
                                    
                }
                else if (detectedObject.Type == "10um" && !ObjectIsInObjectiveLocation(Objective2CaptureLocation, detectedObject))
                {
                    CreateCaptureLocation(detectedObject, Class2CaptureLocations, Objective2CaptureLocation, AutomatedLightSettingsDefaultValues[1]);
                }
              
            }
        }

        private bool ObjectIsInObjectiveLocation(CaptureObjectCheckBox ObjectiveCaptureLocation,DetectedObject detectedObject)
        {
            if (ObjectiveCaptureLocation != null)
            {
                var upperleftPoint = new Point(ObjectiveCaptureLocation.captureLocation.Location.X - ObjectiveCaptureLocation.captureLocation.shape.RectangleObject.Width / 2, ObjectiveCaptureLocation.captureLocation.Location.Y - ObjectiveCaptureLocation.captureLocation.shape.RectangleObject.Height / 2);
                var lowerRightPoint = new Point(upperleftPoint.X + ObjectiveCaptureLocation.captureLocation.shape.RectangleProps.Width, upperleftPoint.Y + ObjectiveCaptureLocation.captureLocation.shape.RectangleProps.Height);
                if (detectedObject.X <= lowerRightPoint.X && detectedObject.X >= upperleftPoint.X && detectedObject.Y <= lowerRightPoint.Y && detectedObject.Y >= upperleftPoint.Y)
                {
                    return true;
                }
            }
       
            return false;

        }

        void updateLinearMovemnets()
        {
            if (cbEnableTransport.Checked)
            {
                foreach (var cb in Class1CaptureLocations)
                {
                    if (cb!=null && Objective1CaptureLocation != null)
                    {
                        cb.updateLinearMovement(new LinearMovement(cb, cb.Location, Objective1CaptureLocation.captureLocation.Location, Properties.Misc.Default.TranslationRate, Properties.Misc.Default.MicrometerPerPixel));
                    }
                }
                foreach (var capture in Class2CaptureLocations)
                {
                    if (capture!=null && Objective2CaptureLocation != null)
                    {
                        capture.updateLinearMovement(new LinearMovement(capture, capture.Location, Objective2CaptureLocation.captureLocation.Location, Properties.Misc.Default.TranslationRate, Properties.Misc.Default.MicrometerPerPixel));
                    }
                }
            }
        }

        private void CreateCaptureLocation(DetectedObject detectedObject,List<CaptureLocation> captureLocations,CaptureObjectCheckBox ObjectiveCaptureLocation,ShapeProperties shapeProperties)
        {

            var relativeLocation = new Point(detectedObject.X, detectedObject.Y);

            var captureLocation = new CaptureLocation(relativeLocation, shapeProperties);
            
            captureLocation.ObjectDetectionID = detectedObject.ID;
            
            MainForm.CreateCaptureLocation(new Point(detectedObject.X, detectedObject.Y), true, captureLocation);
            MainForm.AutomatedCaptureCheckboxList.Add(captureLocation.Parent);
            captureLocations.Add(captureLocation);
            captureLocation.OnFinishedTranslation += CpLocation_OnFinishedTranslation;
            
            if (ObjectiveCaptureLocation != null && cbEnableTransport.Checked)
            {
                captureLocation.updateLinearMovement(new LinearMovement(captureLocation, captureLocation.Location, ObjectiveCaptureLocation.captureLocation.Location, Properties.Misc.Default.TranslationRate, Properties.Misc.Default.MicrometerPerPixel)); ;
               
            }
            OnCaptureLocationCreatedHandler();
        }

       
        public event EventHandler OnCaptureLocationCreated;
        protected virtual void OnCaptureLocationCreatedHandler()
        {
            if (OnCaptureLocationCreated != null)
            {
                OnCaptureLocationCreated(this, null);
            }
        }

        private void CpLocation_OnFinishedTranslation(object sender, EventArgs e)
        {
            var obj = sender as CaptureLocation;
            if (obj != null)
            {
                obj.Parent.OnRemoveClicked();
            }
        }

        private bool checkIfRingExists(int X, int Y, int Width, int Height,string type)
        {
            var Padding = 15;
            foreach (var box in MainForm.AutomatedCaptureCheckboxList)
            {
                if (box!=null&& box.captureLocation != null)
                {
                    var Location = box.captureLocation.Location;
                    if (Location.X <= X + (Width / 2+ Padding) && Location.X >= X -( Width / 2+ Padding))
                    {
                        if (Location.Y <= Y + (Height / 2+ Padding) && Location.Y >= Y - (Height / 2 + Padding))
                        {
                            return true;
                        }
                    }
                }
                else { }                        
            }
            return false;
        }
        
        public void ConnectToServer()
        {
            using (TcpClient client = new TcpClient())
            {
                try
                {
                    client.Connect(IPAddress.ToString(), portNumber);
                    MessageBox.Show("Successfully Connected To Server");
                }
                catch(Exception e)
                {
                    MessageBox.Show(e.Message);
                }
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            ConnectToServer();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (ServerListening)
            {
                ServerListening = false;
                groupBox1.Enabled = true;
                StopServer = true;
                ObjectDetectionServerObject.Stop();
                ObjectDetectionServerObject = null;
                ServerListening = false;
                button2.Text = "Start Server";
                button3.Enabled = false;
                label4.BackColor = Color.Red;
                UpdateDataReadOutTextbox("Server Stopped");
            }
            else
            {
                groupBox1.Enabled = false;
                StopServer = false;
                button2.Text = "Stop Server";
                button3.Enabled = true;
                BeginListening();
                label4.BackColor = Color.Green;
                UpdateDataReadOutTextbox("Server Started");
            }
            
        }

        private void UpdateDataReadOutTextbox(string data)
        {
            if (tbDataDisplay.InvokeRequired)
            {
                tbDataDisplay.BeginInvoke(new MethodInvoker(delegate { PrintData(); }));
            }
            else 
            {
                PrintData();
            }
            void PrintData()
            {
                tbDataDisplay.Text = data;
            }

        }

        private void button3_Click(object sender, EventArgs e)
        {
            button2_Click(null, null);
            button2_Click(null, null);
            UpdateDataReadOutTextbox("Server Restarted");
        }

        private void button4_Click(object sender, EventArgs e)
        {
            StopServer = true;
            this.FormClosing -= ObjectDetectionServer_FormClosing;
            if (ObjectDetectionServerObject != null)
            {                
                ObjectDetectionServerObject.Stop();
            }
            this.Close();
        }

        private void button5_Click(object sender, EventArgs e)
        {
            this.WindowState = FormWindowState.Minimized;
        }

        

        private void settingsButton2_Click(object sender, EventArgs e)
        {
            Capture_Properties_Form capture_Properties_Form = new Capture_Properties_Form(AutomatedLightSettingsDefaultValues[0]);
            var result=capture_Properties_Form.ShowDialog();
            AutomatedLightSettingsDefaultValues[0] = capture_Properties_Form.ShapePropertiesForAutomatedCaptureLocations;
            foreach (var captureLocation in Class1CaptureLocations)
            {
                if (captureLocation != null)
                {
                    captureLocation.shape = AutomatedLightSettingsDefaultValues[0];
                }
            }
        }

        private void settingsButton1_Click(object sender, EventArgs e)
        {
            Capture_Properties_Form capture_Properties_Form = new Capture_Properties_Form(AutomatedLightSettingsDefaultValues[1]);
            var result = capture_Properties_Form.ShowDialog();
            AutomatedLightSettingsDefaultValues[1] = capture_Properties_Form.ShapePropertiesForAutomatedCaptureLocations;
            foreach (var captureLocation in Class2CaptureLocations)
            {
                if (captureLocation != null)
                {
                    captureLocation.shape = AutomatedLightSettingsDefaultValues[1];
                }
                
            }
        }

        private CaptureObjectCheckBox Objective1CaptureLocation;
        private CaptureObjectCheckBox Objective2CaptureLocation;
        private void button5_Click_1(object sender, EventArgs e)
        {
            Objective1CaptureLocation=ClassObjectiveCaptureLocation(tbObjective1Name);            
        }

        private void button6_Click(object sender, EventArgs e)
        {
            Objective2CaptureLocation=ClassObjectiveCaptureLocation(tbObjective2Name);
        }

        private CaptureObjectCheckBox ClassObjectiveCaptureLocation(TextBox tb)
        {
            CaptureLocationSelectionForm captureLocationSelectionForm = new CaptureLocationSelectionForm(MainForm.captureLocationSelections);
            var result = captureLocationSelectionForm.ShowDialog();
            var capture = captureLocationSelectionForm.locationSelectedFromList as CaptureObjectCheckBox;
            
            if (capture != null)
            {
                foreach (var obj in MainForm.captureLocationSelections)
                {
                    if (obj.captureLocation.Equals(capture.captureLocation))
                    {
                        capture = obj;
                    }
                }
                capture.TextChanged += Capture_TextChanged;
                tb.Text=capture.Text;
                return capture ;
            }
            return null;
        }

        private void Capture_TextChanged(object sender, EventArgs e)
        {
            var tb = (sender as CaptureObjectCheckBox);
            if (tb.Equals(Objective1CaptureLocation))
            {
                tbObjective1Name.Text = Objective1CaptureLocation.Text;
            }
            else if(tb.Equals(Objective2CaptureLocation))
            {
                tbObjective2Name.Text = Objective2CaptureLocation.Text;
            }
        }

        private void button1_Click_1(object sender, EventArgs e)
        {
            updateLinearMovemnets();
        }
        public static bool CheckForCollusion(CaptureLocation captureLocation)
        {
            if (CheckCollusionInList(captureLocation, Class1CaptureLocations))
            {
                return true;
            }
            if (CheckCollusionInList(captureLocation, Class2CaptureLocations))
            {
                return true;
            }
            return false;
        }

        private static bool CheckCollusionInList(CaptureLocation captureLocation,List<CaptureLocation> ObjectDetectionList)
        {
            Beginning:
            for (int index =0;index< ObjectDetectionList.Count;index++) 
            {
                var obj = ObjectDetectionList[index];
                if (obj!=null&&!obj.Equals(captureLocation))
                {                    
                    if (obj.Parent==null||obj.Parent.IsDisposed)
                    {
                        try
                        {
                            ObjectDetectionList.Remove(obj);
                        }
                        catch(System.ArgumentOutOfRangeException ex)
                        {

                        }
                        catch(Exception e)
                        {
                            Console.WriteLine(e.Message)
                        }
                        goto Beginning;
                    }
                    if (SeperationDistance(obj.Location, captureLocation.Location) < distanceToMaintain && captureLocation.ObjectDetectionID > obj.ObjectDetectionID)
                    {
                        return true;
                    }
                }
            }
            return false;

        }
        
        private static double SeperationDistance(Point point1,Point point2)
        {
            var DelX = (point2.X - point1.X);
            var DelY = (point2.Y - point1.Y);
            var Angle = Math.Atan2(DelY, DelX);
            return Math.Sqrt(((DelX * DelX) + (DelY * DelY)));
        }
        
        private void textBox1_TextChanged_1(object sender, EventArgs e)
        {
            var tb = sender as TextBox;
            if(int.TryParse(tb.Text,out int number))
            {
                distanceToMaintain = number;
            }
        }

        private void tbULX_TextChanged(object sender, EventArgs e)
        {
            var tb = (sender as InputTextBox);
            if(int.TryParse(tb.Text,out int value))
            {
                var settings = Properties.AutomationSettings.Default;
                switch (tb.TypeOfInput)
                {
                    case InputTextBox.InputTypeEnum.Upper_LeftX:
                        settings.UpperLeftCorner = new Point(value, settings.UpperLeftCorner.Y);
                        break;
                    case InputTextBox.InputTypeEnum.Upper_LeftY:
                        settings.UpperLeftCorner = new Point(settings.UpperLeftCorner.X, value);
                        break;
                    case InputTextBox.InputTypeEnum.Lower_RightX:
                        settings.LowerRightCorner = new Point(value, settings.LowerRightCorner.Y);
                        break;
                    case InputTextBox.InputTypeEnum.Lower_RightY:
                        settings.LowerRightCorner = new Point(settings.LowerRightCorner.X, value);
                        break;
                }
                settings.Save();
            }
        }
    }

    public class DetectedObject
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public string Type { get; set; }
        
        public int ID { get; set; }

        public int Xlimit { get; set; }
        public int Ylimit { get; set; }
        public DetectedObject()
        {
            SetID();
        }

        public void SetID()
        {
            ID = Properties.Settings.Default.DetectedObjectIndex;
            Properties.Settings.Default.DetectedObjectIndex++;
        }
    }

    class InputTextBox:TextBox
    {
        public enum InputTypeEnum {Upper_LeftX, Upper_LeftY, Lower_RightX, Lower_RightY};
        public InputTypeEnum TypeOfInput { get; set; }
    }
   
}
