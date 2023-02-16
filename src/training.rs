use std::cmp;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

use bert_tokenizer::{FullTokenizer, Tokenizer};

use rand::prelude::*;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tfrecord::{Example, ExampleWriter, Feature};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrainingInstance2D {
    pub tokens: Vec<String>,
    pub segment_ids: Vec<u8>,
    pub masked_lm_positions: Vec<u16>,
    pub masked_lm_labels: Vec<String>,
    pub is_random_next: bool,
}

impl TrainingInstance2D {
    pub fn new(
        tokens: Vec<String>,
        segment_ids: Vec<u8>,
        masked_lm_positions: Vec<u16>,
        masked_lm_labels: Vec<String>,
        is_random_next: bool,
    ) -> Self {
        Self {
            tokens,
            segment_ids,
            masked_lm_positions,
            masked_lm_labels,
            is_random_next,
        }
    }
}

fn create_int_feature<T>(values: Vec<T>) -> Feature
where
    T: Into<u64>,
{
    Feature::from_i64_iter(values.into_iter().map(|x| x.into() as i64))
}

fn create_float_feature(values: Vec<f32>) -> Feature {
    Feature::from_f32_list(values)
}

pub fn write_instance_to_example_files(
    instances: Vec<TrainingInstance2D>,
    tokenizer: FullTokenizer,
    max_seq_length: u16,
    max_predictions_per_seq: u16,
    output_files: Vec<String>,
) {
    let mut writers: Vec<ExampleWriter<_>> = output_files
        .into_iter()
        .map(ExampleWriter::create)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let mut writer_index = 0;

    for (instance_index, instance) in instances.iter().enumerate() {
        let tokens = instance.tokens.clone();
        let mut input_ids = tokenizer.convert_tokens_to_ids(&tokens);
        let mut input_mask = vec![1u8; input_ids.len()];
        let mut segment_ids = instance.segment_ids.clone();

        while (input_ids.len() as u16) < max_seq_length {
            input_ids.push(0);
            input_mask.push(0);
            segment_ids.push(0);
        }

        assert_eq!(input_ids.len(), max_seq_length as usize);
        assert_eq!(input_mask.len(), max_seq_length as usize);
        assert_eq!(segment_ids.len(), max_seq_length as usize);

        let mut masked_lm_positions = instance.masked_lm_positions.clone();
        let masked_lm_labels = instance.masked_lm_labels.clone();
        let mut masked_lm_ids = tokenizer.convert_tokens_to_ids(&masked_lm_labels);
        let mut masked_lm_weights = vec![1.0; masked_lm_ids.len()];

        while masked_lm_positions.len() < max_predictions_per_seq as usize {
            masked_lm_positions.push(0);
            masked_lm_ids.push(0);
            masked_lm_weights.push(0.0);
        }

        assert_eq!(masked_lm_positions.len(), max_predictions_per_seq as usize);
        assert_eq!(masked_lm_ids.len(), max_predictions_per_seq as usize);
        assert_eq!(masked_lm_weights.len(), max_predictions_per_seq as usize);

        let next_sentence_label = match instance.is_random_next {
            true => 1u8,
            false => 0u8,
        };

        let example = vec![
            ("input_ids".into(), create_int_feature(input_ids)),
            ("input_mask".into(), create_int_feature(input_mask)),
            ("segment_ids".into(), create_int_feature(segment_ids)),
            (
                "masked_lm_positions".into(),
                create_int_feature(masked_lm_positions),
            ),
            ("masked_lm_ids".into(), create_int_feature(masked_lm_ids)),
            (
                "masked_lm_weights".into(),
                create_float_feature(masked_lm_weights),
            ),
            (
                "next_sentence_labels".into(),
                create_int_feature(vec![next_sentence_label]),
            ),
        ]
        .into_iter()
        .collect::<Example>();

        if instance_index < 20 {
            println!("*** Example ***");
            println!("tokens: {:?}", instance.tokens);
            for feature in &example.features {
                println!("{feature:?}");
            }
        }

        writers[writer_index].send(example).unwrap();
        writer_index = (writer_index + 1) % writers.len();
    }
}

pub struct TrainingInstanceCreator<'a> {
    rng: &'a mut ChaCha8Rng,
    tokenizer: &'a FullTokenizer,
    max_seq_length: u16,
    dupe_factor: u8,
    short_seq_prob: f32,
    masked_lm_prob: f32,
    do_whole_word_masking: bool,
    max_predictions_per_seq: u16,
}

struct MaskedLmPrediction {
    masked_tokens: Vec<String>,
    masked_lm_positions: Vec<u16>,
    masked_lm_labels: Vec<String>,
}

struct MaskedLmInstance {
    index: u16,
    label: String,
}

pub struct TrainingArgs {
    max_seq_length: u16,
    dupe_factor: u8,
    short_seq_prob: f32,
    masked_lm_prob: f32,
    do_whole_word_masking: bool,
    max_predictions_per_seq: u16,
}

impl TrainingArgs {
    pub fn new(
        max_seq_length: u16,
        dupe_factor: u8,
        short_seq_prob: f32,
        masked_lm_prob: f32,
        do_whole_word_masking: bool,
        max_predictions_per_seq: u16,
    ) -> Self {
        Self {
            max_seq_length,
            dupe_factor,
            short_seq_prob,
            masked_lm_prob,
            do_whole_word_masking,
            max_predictions_per_seq,
        }
    }
}

impl<'a> TrainingInstanceCreator<'a> {
    pub(crate) fn new(
        rng: &'a mut ChaCha8Rng,
        tokenizer: &'a FullTokenizer,
        training_args: TrainingArgs,
    ) -> TrainingInstanceCreator<'a> {
        Self {
            rng,
            tokenizer,
            max_seq_length: training_args.max_seq_length,
            dupe_factor: training_args.dupe_factor,
            short_seq_prob: training_args.short_seq_prob,
            masked_lm_prob: training_args.masked_lm_prob,
            do_whole_word_masking: training_args.do_whole_word_masking,
            max_predictions_per_seq: training_args.max_predictions_per_seq,
        }
    }

    pub fn create_training_instances(
        &mut self,
        input_file_names: Vec<String>,
    ) -> Vec<TrainingInstance2D> {
        let mut all_documents: Vec<Vec<Vec<String>>> = vec![];

        // Read documents from input files
        for input_file in input_file_names {
            let mut reader = BufReader::new(File::open(input_file).unwrap());
            let mut line = String::new();
            let mut current_document: Vec<Vec<String>> = vec![];

            while reader.read_line(&mut line).unwrap() > 0 {
                let trimmed_line = line.trim().to_owned();
                if trimmed_line.is_empty() {
                    if !current_document.is_empty() {
                        all_documents.push(current_document);
                        current_document = vec![];
                    }
                } else {
                    let tokens = self.tokenizer.tokenize(&trimmed_line);
                    current_document.push(tokens);
                }
                line.clear();
            }

            if !current_document.is_empty() {
                all_documents.push(current_document);
            }
        }

        all_documents.shuffle(&mut self.rng);

        let vocab_words = self.tokenizer.get_vocab_words();
        let mut instances = vec![];

        for _ in 0..self.dupe_factor {
            for document_index in 0..all_documents.len() {
                let created_instance = self.create_instance_from_document(
                    &all_documents,
                    document_index,
                    &vocab_words,
                );
                instances.extend(created_instance);
            }
        }

        instances.shuffle(&mut self.rng);

        instances
    }

    fn create_instance_from_document(
        &mut self,
        all_documents: &Vec<Vec<Vec<String>>>,
        document_index: usize,
        vocab_words: &Vec<String>,
    ) -> Vec<TrainingInstance2D> {
        let document = &all_documents[document_index];
        let max_num_tokens = self.max_seq_length - 3;

        let target_seq_length: u16 = if self.rng.gen_bool(self.short_seq_prob as f64) {
            self.rng.gen_range(2u16..max_num_tokens)
        } else {
            max_num_tokens
        };

        let mut instances = vec![];
        let mut current_chunk: Vec<Vec<String>> = vec![];
        let mut current_length = 0u16;
        let mut i = 0;

        while i < document.len() {
            let segment = &document[i];
            current_chunk.push(segment.to_vec());
            current_length += segment.len() as u16;
            if i == document.len() - 1 || current_length >= target_seq_length {
                if !current_chunk.is_empty() {
                    let instance = self.create_instance(
                        &mut i,
                        &current_chunk,
                        target_seq_length,
                        document_index,
                        all_documents,
                        vocab_words,
                    );
                    instances.push(instance);
                }
                current_chunk = vec![];
                current_length = 0;
            }
            i += 1;
        }

        instances
    }

    fn create_instance(
        &mut self,
        i: &mut usize,
        current_chunk: &Vec<Vec<String>>,
        target_seq_length: u16,
        document_index: usize,
        all_documents: &Vec<Vec<Vec<String>>>,
        vocab_words: &Vec<String>,
    ) -> TrainingInstance2D {
        let mut a_end = 1;
        let current_chunk_len = current_chunk.len();
        if current_chunk_len >= 2 {
            a_end = self.rng.gen_range(1..current_chunk_len);
        }
        let mut tokens_a = current_chunk[0..a_end]
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<String>>();
        let mut is_random_next = false;
        let mut tokens_b = if current_chunk_len > 1 && !self.rng.gen_bool(0.5) {
            current_chunk[a_end..current_chunk_len]
                .iter()
                .flatten()
                .cloned()
                .collect::<Vec<String>>()
        } else {
            is_random_next = true;
            let target_b_length = if target_seq_length as usize > tokens_a.len() {
                target_seq_length as usize - tokens_a.len()
            } else {
                0
            };
            let random_document_index =
                self.get_random_document_index(all_documents.len(), document_index);
            let random_document = &all_documents[random_document_index];
            let random_start = self.rng.gen_range(0..random_document.len());
            let mut tokens_b = vec![];
            for random_segment in random_document.iter().skip(random_start) {
                tokens_b.extend(random_segment.clone());
                if tokens_b.len() >= target_b_length {
                    break;
                }
            }
            let num_unused_segments = current_chunk_len - a_end;
            *i -= num_unused_segments;
            tokens_b
        };

        self.truncate_seq_pair(&mut tokens_a, &mut tokens_b, target_seq_length);

        let mut tokens = vec![];
        let mut segment_ids = vec![];

        tokens.push("[CLS]".to_owned());
        segment_ids.push(0);

        for token in tokens_a {
            tokens.push(token);
            segment_ids.push(0);
        }

        tokens.push("[SEP]".to_owned());
        segment_ids.push(0);

        for token in tokens_b {
            tokens.push(token);
            segment_ids.push(1);
        }

        tokens.push("[SEP]".to_owned());
        segment_ids.push(1);

        let masked_lm_prediction = self.create_masked_lm_predictions(&tokens, vocab_words);

        TrainingInstance2D::new(
            masked_lm_prediction.masked_tokens,
            segment_ids,
            masked_lm_prediction.masked_lm_positions,
            masked_lm_prediction.masked_lm_labels,
            is_random_next,
        )
    }

    fn get_random_document_index(
        &mut self,
        all_documents_len: usize,
        document_index: usize,
    ) -> usize {
        if all_documents_len == 1 {
            return 0;
        }

        let mut random_document_index = document_index;

        for _ in 0..10 {
            random_document_index = self.rng.gen_range(0..all_documents_len);
            if random_document_index != document_index {
                break;
            }
        }

        random_document_index
    }

    fn truncate_seq_pair(
        &mut self,
        tokens_a: &mut Vec<String>,
        tokens_b: &mut Vec<String>,
        target_seq_length: u16,
    ) {
        loop {
            let total_length = tokens_a.len() + tokens_b.len();
            if total_length <= target_seq_length as usize {
                break;
            }

            self.truncate_seq(if tokens_a.len() > tokens_b.len() {
                tokens_a
            } else {
                tokens_b
            });
        }
    }

    fn truncate_seq(&mut self, tokens: &mut Vec<String>) {
        if self.rng.gen_bool(0.5) {
            tokens.remove(0);
        } else {
            tokens.pop();
        }
    }

    fn create_masked_lm_predictions(
        &mut self,
        tokens: &Vec<String>,
        vocab_words: &Vec<String>,
    ) -> MaskedLmPrediction {
        let mut cand_indexes: Vec<Vec<usize>> = vec![];
        for (i, token) in tokens.iter().enumerate() {
            if token == "[CLS]" || token == "[SEP]" {
                continue;
            }
            if self.do_whole_word_masking && !cand_indexes.is_empty() && token.starts_with("##") {
                cand_indexes.last_mut().unwrap().push(i);
            } else {
                cand_indexes.push(vec![i]);
            }
        }

        cand_indexes.shuffle(&mut self.rng);

        let num_to_predict = cmp::min(
            self.max_predictions_per_seq,
            cmp::max(
                1,
                (tokens.len() as f32 * self.masked_lm_prob).round() as u16,
            ),
        );

        let mut masked_lms = vec![];
        let mut covered_indexes = HashSet::new();
        let mut masked_tokens = tokens.clone();

        for index_set in cand_indexes {
            if masked_lms.len() >= num_to_predict as usize {
                break;
            }

            if masked_lms.len() + index_set.len() > num_to_predict as usize {
                continue;
            }

            let mut is_any_index_covered = false;
            for index in &index_set {
                if covered_indexes.contains(index) {
                    is_any_index_covered = true;
                    break;
                }
            }

            if is_any_index_covered {
                continue;
            }

            for index in index_set {
                covered_indexes.insert(index);

                let masked_token = if self.rng.gen_bool(0.8) {
                    "[MASK]".to_owned()
                } else if self.rng.gen_bool(0.5) {
                    tokens[index].clone()
                } else {
                    vocab_words[self.rng.gen_range(0..vocab_words.len())].clone()
                };

                masked_tokens[index] = masked_token;
                masked_lms.push(MaskedLmInstance {
                    index: index as u16,
                    label: tokens[index].clone(),
                });
            }
        }

        masked_lms.sort_by(|a, b| a.index.cmp(&b.index));

        let (masked_lm_positions, masked_lm_labels) = masked_lms
            .into_iter()
            .map(|masked_lm| (masked_lm.index, masked_lm.label))
            .unzip();

        MaskedLmPrediction {
            masked_tokens,
            masked_lm_positions,
            masked_lm_labels,
        }
    }
}
