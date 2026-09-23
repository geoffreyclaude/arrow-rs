// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::hint;

use arrow_arith::numeric::{add, div, mul, rem, sub};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    Decimal32Type, Decimal64Type, Decimal128Type, Decimal256Type, DecimalType,
};
use arrow_array::{Array, PrimitiveArray};
use arrow_buffer::ArrowNativeType;
use arrow_schema::ArrowError;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

// Benchmark-local baseline for the scalar API. The scalar implementation commit
// replaces this array-backed adapter with the allocation-free public API.
#[derive(Clone, Copy)]
enum DecimalOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

fn decimal_scalar<T: DecimalType>(
    op: DecimalOp,
    left: (Option<T::Native>, u8, i8),
    right: (Option<T::Native>, u8, i8),
) -> Result<(Option<T::Native>, u8, i8), ArrowError> {
    let left = PrimitiveArray::<T>::from_iter([left.0]).with_precision_and_scale(left.1, left.2)?;
    let right =
        PrimitiveArray::<T>::from_iter([right.0]).with_precision_and_scale(right.1, right.2)?;
    let result = match op {
        DecimalOp::Add => add(&left, &right),
        DecimalOp::Sub => sub(&left, &right),
        DecimalOp::Mul => mul(&left, &right),
        DecimalOp::Div => div(&left, &right),
        DecimalOp::Rem => rem(&left, &right),
    }?;
    let result = result.as_primitive::<T>();
    Ok((
        (!result.is_null(0)).then(|| result.value(0)),
        result.precision(),
        result.scale(),
    ))
}

macro_rules! decimal_operation {
    ($group:ident, $type:ty, $name:expr, $op:expr, $kernel:ident, $left:expr, $right:expr) => {{
        let left = $left;
        let right = $right;
        $group.bench_function(format!("scalar/{}", $name), |b| {
            b.iter(|| {
                hint::black_box(
                    decimal_scalar::<$type>($op, hint::black_box(left), hint::black_box(right))
                        .unwrap(),
                )
            })
        });
        $group.bench_function(format!("array_construct/{}", $name), |b| {
            b.iter(|| {
                let left = PrimitiveArray::<$type>::from_iter([hint::black_box(left.0)])
                    .with_precision_and_scale(left.1, left.2)
                    .unwrap();
                let right = PrimitiveArray::<$type>::from_iter([hint::black_box(right.0)])
                    .with_precision_and_scale(right.1, right.2)
                    .unwrap();
                hint::black_box($kernel(&left, &right).unwrap())
            })
        });
        let left_array = PrimitiveArray::<$type>::from_iter([left.0])
            .with_precision_and_scale(left.1, left.2)
            .unwrap();
        let right_array = PrimitiveArray::<$type>::from_iter([right.0])
            .with_precision_and_scale(right.1, right.2)
            .unwrap();
        $group.bench_function(format!("array_prebuilt/{}", $name), |b| {
            b.iter(|| hint::black_box($kernel(&left_array, &right_array).unwrap()))
        });
    }};
}

macro_rules! decimal_scenario {
    ($group:ident, $type:ty, $scenario:literal, $left:expr, $right:expr) => {
        decimal_operation!(
            $group,
            $type,
            concat!($scenario, "/add"),
            DecimalOp::Add,
            add,
            $left,
            $right
        );
        decimal_operation!(
            $group,
            $type,
            concat!($scenario, "/sub"),
            DecimalOp::Sub,
            sub,
            $left,
            $right
        );
        decimal_operation!(
            $group,
            $type,
            concat!($scenario, "/mul"),
            DecimalOp::Mul,
            mul,
            $left,
            $right
        );
        decimal_operation!(
            $group,
            $type,
            concat!($scenario, "/div"),
            DecimalOp::Div,
            div,
            $left,
            $right
        );
        decimal_operation!(
            $group,
            $type,
            concat!($scenario, "/rem"),
            DecimalOp::Rem,
            rem,
            $left,
            $right
        );
    };
}

fn decimal<T: DecimalType>(c: &mut Criterion, name: &str) {
    let value = |value| Some(T::Native::usize_as(value));
    let mut group = c.benchmark_group(format!("decimal/{name}"));
    group.throughput(Throughput::Elements(1));
    decimal_scenario!(
        group,
        T,
        "equal_scale",
        (value(123), 8, 2),
        (value(7), 7, 2)
    );
    decimal_scenario!(
        group,
        T,
        "different_scale",
        (value(123), 8, 2),
        (value(7), 7, 1)
    );
    decimal_operation!(
        group,
        T,
        "null/div",
        DecimalOp::Div,
        div,
        (None, 8, 2),
        (value(7), 7, 1)
    );
    group.finish();
}

fn decimals(c: &mut Criterion) {
    decimal::<Decimal32Type>(c, "decimal32");
    decimal::<Decimal64Type>(c, "decimal64");
    decimal::<Decimal128Type>(c, "decimal128");
    decimal::<Decimal256Type>(c, "decimal256");
}

criterion_group!(benches, decimals);
criterion_main!(benches);
