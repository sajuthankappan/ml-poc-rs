use csv::ReaderBuilder;
use linfa::{
    prelude::{Records, ToConfusionMatrix},
    traits::{Fit, Predict},
    Dataset,
};
use linfa_logistic::MultiLogisticRegression;
use ndarray::{array, s, Array2, Ix1};
use ndarray_csv::Array2Reader;

use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let train = load_data("data/train.csv");
    let test = load_data("data/test.csv");

    let features = train.nfeatures();
    let targets = train.ntargets();

    println!(
        "training with {} samples, testing with {} samples, {} features and {} target",
        train.nsamples(),
        test.nsamples(),
        features,
        targets
    );

    let model = MultiLogisticRegression::default()
        .max_iterations(1000)
        .fit(&train)
        .expect("cannot train model");

    let pred = model.predict(&test);

    let cm = pred.confusion_matrix(&test).unwrap();

    println!("cm {:?}", &cm);

    println!("accuracy {}, MCC {}", &cm.accuracy(), &cm.mcc());

    let final_pred_array = array![[20.0, 0.0]];
    let final_pred_dataset = convert_array_to_dataset(final_pred_array);
    let final_pred = model.predict(&final_pred_dataset);
    dbg!(&final_pred.get(0));

    Ok(())
}

fn load_data(path: &str) -> Dataset<f64, &'static str, Ix1> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .from_path(path)
        .expect("can't create reader");

    let array: Array2<f64> = reader
        .deserialize_array2_dynamic()
        .expect("can't deserialize array");

    let ds = convert_array_to_dataset(array);
    ds
}

fn convert_array_to_dataset(array: Array2<f64>) -> Dataset<f64, &'static str, Ix1> {
    let (data, targets) = (
        array.slice(s![.., 0..1]).to_owned(),
        array.column(1).to_owned(),
    );
    let feature_names = vec!["marks"];

    Dataset::new(data, targets)
        .map_targets(|x| {
            if *x as f64 == 1.0 {
                "pass"
            } else if *x as f64 == 0.5 {
                "half"
            } else {
                "fail"
            }
        })
        .with_feature_names(feature_names)
}
