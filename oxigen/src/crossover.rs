//! This module contains the Crossover trait and the provided crossover functions.

use genotype::Genotype;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::cmp::{min, PartialEq};
// use std::mem::replace;

use CrossoverFunctions::*;

/// This trait defines the cross function.
pub trait Crossover<T: PartialEq, G: Genotype<T>>: Send + Sync {
    /// Generates two children combining the two selected individuals.
    fn cross(&self, ind1: &G, ind2: &G) -> (G, G);
}

/// Provided crossover functions.
#[derive(Debug)]
pub enum CrossoverFunctions {
    /// Single point Crossover.
    SingleCrossPoint,
    /// Multi point Crossover.
    MultiCrossPoint,
    /// Uniform Crossover.
    UniformCross, /*
                  /// Uniform Partially Matched
                  UniformPartiallyMatched(f32),
                  /// Partially Matched
                  PartiallyMatched*/
}

impl<T: PartialEq, G: Genotype<T>> Crossover<T, G> for CrossoverFunctions {
    fn cross(&self, ind1: &G, ind2: &G) -> (G, G) {
        match self {
            SingleCrossPoint => {
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                let cross_point = SmallRng::from_entropy().sample(Uniform::from(1..ind_size));

                let mut child1 = ind1.clone();
                child1.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| if i < cross_point { gen1 } else { gen2 }),
                );
                let mut child2 = ind2.clone();
                child2.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| if i < cross_point { gen2 } else { gen1 }),
                );

                (child1, child2)
            }
            MultiCrossPoint => {
                let ind_size = min(ind1.iter().len(), ind2.iter().len());
                let mut cross_points = Vec::new();
                let mut point_maximum = ind_size / 2;
                if point_maximum <= 2 {
                    point_maximum = 3;
                }
                let mut i = SmallRng::from_entropy().sample(Uniform::from(1..point_maximum));
                while i < ind_size {
                    cross_points.push(i);
                    i += SmallRng::from_entropy().sample(Uniform::from(1..point_maximum));
                }
                cross_points.push(ind_size);

                let mut child1 = ind1.clone();
                child1.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| {
                            let mut even = false;
                            for cross_point in &cross_points {
                                if i < *cross_point {
                                    if even {
                                        return gen2;
                                    } else {
                                        return gen1;
                                    }
                                } else {
                                    even = !even;
                                }
                            }
                            gen1
                        }),
                );
                let mut child2 = ind2.clone();
                child2.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| {
                            let mut even = false;
                            for cross_point in &cross_points {
                                if i < *cross_point {
                                    if even {
                                        return gen1;
                                    } else {
                                        return gen2;
                                    }
                                } else {
                                    even = !even;
                                }
                            }
                            gen2
                        }),
                );

                (child1, child2)
            }
            UniformCross => {
                let mut child1 = ind1.clone();
                child1.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| if i % 2 == 0 { gen1 } else { gen2 }),
                );
                let mut child2 = ind2.clone();
                child2.from_iter(
                    ind1.clone()
                        .into_iter()
                        .zip(ind2.clone().into_iter())
                        .enumerate()
                        .map(|(i, (gen1, gen2))| if i % 2 != 0 { gen1 } else { gen2 }),
                );

                (child1, child2)
            } /*
              UniformPartiallyMatched(indpb) => {
                  let size = min(ind1.iter().len(), ind2.iter().len());

                  let mut ind1_temp : Vec<Option<T>> = ind1.clone().into_iter().map(|e| Some(e)).collect();
                  let mut ind2_temp : Vec<Option<T>> = ind2.clone().into_iter().map(|e| Some(e)).collect();

                  let mut i1 : Vec<usize> = (0..ind1_temp.len()).collect();
                  let mut i2 : Vec<usize> = (0..ind2_temp.len()).collect();

                  let mut p1 = vec![0; size];
                  let mut p2 = vec![0; size];

                  for i in 0..size {
                      p1[i1[i]] = i;
                      p2[i2[i]] = i;
                  }

                  for i in 0..size {
                      let p : f32 = SmallRng::from_entropy().gen();
                      if p < *indpb {
                          let temp1 = i1[i];
                          let temp2 = i2[i];

                          i1[i] = temp2;
                          i1[p1[temp2]] = temp1;

                          i2[i] = temp1;
                          i2[p2[temp1]] = temp2;

                          p1[temp1] = p1[temp2];
                          p1[temp2] = p1[temp1];
                          p2[temp1] = p2[temp2];
                          p2[temp2] = p2[temp1];
                      }
                  }

                  let mut child1 = ind1.clone();
                  child1.from_iter(
                      i1.iter().map(|loc| {
                          replace(&mut ind1_temp[*loc], None).unwrap()
                      })
                  );

                  let mut child2 = ind2.clone();
                  child2.from_iter(
                      i2.iter().map(|loc| {
                          replace(&mut ind2_temp[*loc], None).unwrap()
                      })
                  );

                  (child1, child2)
              }

              PartiallyMatched => {
                  let size = min(ind1.iter().len(), ind2.iter().len());

                  let mut ind1_temp : Vec<Option<T>> = ind1.clone().into_iter().map(|e| Some(e)).collect();
                  let mut ind2_temp : Vec<Option<T>> = ind2.clone().into_iter().map(|e| Some(e)).collect();

                  let mut i1 : Vec<usize> = (0..ind1_temp.len()).collect();
                  let mut i2 : Vec<usize> = (0..ind2_temp.len()).collect();

                  let mut p1 = vec![0; size];
                  let mut p2 = vec![0; size];

                  for i in 0..size {
                      p1[i1[i]] = i;
                      p2[i2[i]] = i;
                  }

                  // choose crossover points
                  let mut cxpoint1 = SmallRng::from_entropy().sample(Uniform::from(1..size));
                  let mut cxpoint2 = SmallRng::from_entropy().sample(Uniform::from(1..size-1));

                  if cxpoint2 >= cxpoint1 {
                      cxpoint2 += 1;
                  } else {
                      let temp = cxpoint1;
                      cxpoint1 = cxpoint2;
                      cxpoint2 = temp;
                  }

                  // crossover between points
                  for i in cxpoint1..cxpoint2 {
                      let temp1 = i1[i];
                      let temp2 = i2[i];

                      i1[i] = temp2;
                      i1[p1[temp2]] = temp1;

                      i2[i] = temp1;
                      i2[p2[temp1]] = temp2;

                      p1[temp1] = p1[temp2];
                      p1[temp2] = p1[temp1];
                      p2[temp1] = p2[temp2];
                      p2[temp2] = p2[temp1];
                  }

                  let mut child1 = ind1.clone();
                  child1.from_iter(
                      i1.iter().map(|loc| {
                          replace(&mut ind1_temp[*loc], None).unwrap()
                      })
                  );

                  let mut child2 = ind2.clone();
                  child2.from_iter(
                      i2.iter().map(|loc| {
                          replace(&mut ind2_temp[*loc], None).unwrap()
                      })
                  );

                  (child1, child2)
              }*/
        }
    }
}
