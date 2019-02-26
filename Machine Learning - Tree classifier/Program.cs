using System;
using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Transforms.Conversions;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;
using System.IO;
using Microsoft.ML.Core.Data;

namespace Machine_Learning___Tree_classifier
{
    public class TaxiTrip
    {
        [Column("0")]
        public string VendorId;

        [Column("1")]
        public string RateCode;

        [Column("2")]
        public float PassengerCount;

        [Column("3")]
        public float TripTime;

        [Column("4")]
        public float TripDistance;

        [Column("5")]
        public string PaymentType;

        [Column("6")]
        public float FareAmount;

    }
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);
            var pipeline = mlContext.Transforms.CopyColumns("FareAmount", "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                .Append(mlContext.Regression.Trainers.FastTree());
            var model = pipeline.Fit(dataView);
            SaveModelAsFile(mlContext, model);
            return model;
        }
        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);
            Console.WriteLine("The model is saved to {0}", _modelPath);
        }
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
        }
        private static void TestSinglePrediction(MLContext mlContext)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }
            var predictionFunction = loadedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            TaxiTrip[] taxiTripArray =  new TaxiTrip[]
            {
                new TaxiTrip{
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
       
                }
                ,
                new TaxiTrip{
                    VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 480,
                TripDistance = 2.72f,
                PaymentType = "CRD",
                FareAmount = 0  //predicted to be 10
                }
            };

            foreach (var item in taxiTripArray)
            {
                var prediction = predictionFunction.Predict(item);

                Console.WriteLine($"**********************************************************************");
                Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}");
                Console.WriteLine($"**********************************************************************");
            }
        }
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("VendorId", DataKind.Text, 0),
                    new TextLoader.Column("RateCode", DataKind.Text, 1),
                    new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                    new TextLoader.Column("TripTime", DataKind.R4, 3),
                    new TextLoader.Column("TripDistance", DataKind.R4, 4),
                    new TextLoader.Column("PaymentType", DataKind.Text, 5),
                    new TextLoader.Column("FareAmount", DataKind.R4, 6)
                }
            }
            );
            //var model = Train(mlContext, _trainDataPath);
            //Evaluate(mlContext, model);
            TestSinglePrediction(mlContext);
            Console.ReadKey();
        }
    }
   
}
