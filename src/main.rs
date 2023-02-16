use std::time::Instant;

use bert_tokenizer::FullTokenizer;

use anyhow::Result;
use clap::Parser;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::training::{write_instance_to_example_files, TrainingArgs, TrainingInstanceCreator};

mod training;

/// A command line tool for creating pretraining data for BERT.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input raw text file (or comma-separated list of files).
    #[arg(long)]
    input_file: String,

    /// Output TF example file (or comma-separated list of files).
    #[arg(long)]
    output_file: String,

    /// The vocabulary file that the BERT model was trained on.
    #[arg(long)]
    vocab_file: String,

    /// Whether to lower case the input text. Should be True for uncased
    /// models and False for cased models.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    do_lower_case: bool,

    /// Whether to use whole word masking rather than per-WordPiece masking.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    do_whole_word_masking: bool,

    /// Maximum sequence length.
    #[arg(long, default_value = "128")]
    max_seq_length: u16,

    /// Maximum number of masked LM predictions per sequence.
    #[arg(long, default_value = "20")]
    max_predictions_per_seq: u16,

    /// Random number seed for data generation.
    #[arg(long, default_value = "12345")]
    random_seed: u16,

    /// Number of times to duplicate the input data (with different masks).
    /// Maximum is 255.
    #[arg(long, default_value = "10")]
    dupe_factor: u8,

    /// Masked LM probability.
    #[arg(long, default_value = "0.15")]
    masked_lm_prob: f32,

    /// Probability of creating sequences which are shorter than the
    /// maximum length.
    #[arg(long, default_value = "0.10")]
    short_seq_prob: f32,
}

fn main() -> Result<()> {
    let time = Instant::now();
    let args = Args::parse();
    let input_file = args.input_file;
    let output_file = args.output_file;
    let vocab_file = args.vocab_file;
    let do_lower_case = args.do_lower_case;
    let do_whole_word_masking = args.do_whole_word_masking;
    let max_seq_length = args.max_seq_length;
    let max_predictions_per_seq = args.max_predictions_per_seq;
    let random_seed = args.random_seed;
    let dupe_factor = args.dupe_factor;
    let masked_lm_prob = args.masked_lm_prob;
    let short_seq_prob = args.short_seq_prob;

    println!("****** Arguments: ******");
    println!("Input file: {}", &input_file);
    println!("Output file: {}", &output_file);
    println!("Vocab file: {}", &vocab_file);
    println!("Do lower case: {}", &do_lower_case);
    println!("Do whole word masking: {}", &do_whole_word_masking);
    println!("Max sequence length: {}", &max_seq_length);
    println!("Max predictions per sequence: {}", &max_predictions_per_seq);
    println!("Random seed: {}", &random_seed);
    println!("Dupe factor: {}", &dupe_factor);
    println!("Masked LM probability: {}", &masked_lm_prob);
    println!("Short sequence probability: {}", &short_seq_prob);

    let tokenizer = FullTokenizer::new()
        .vocab_from_file(&vocab_file)
        .do_lower_case(do_lower_case)
        .build();
    let mut rng = ChaCha8Rng::seed_from_u64(random_seed as u64);

    let training_args = TrainingArgs::new(
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        masked_lm_prob,
        do_whole_word_masking,
        max_predictions_per_seq,
    );

    let mut instance_creator = TrainingInstanceCreator::new(&mut rng, &tokenizer, training_args);

    let input_file_names = input_file
        .split(',')
        .collect::<Vec<&str>>()
        .into_iter()
        .map(|x| x.to_string())
        .collect();

    let training_instances = instance_creator.create_training_instances(input_file_names);

    let output_file_names = output_file
        .split(',')
        .collect::<Vec<&str>>()
        .into_iter()
        .map(|x| x.to_string())
        .collect();

    write_instance_to_example_files(
        training_instances,
        tokenizer,
        max_seq_length,
        max_predictions_per_seq,
        output_file_names,
    );

    println!("Elapsed: {} ms", time.elapsed().as_millis());

    Ok(())
}
