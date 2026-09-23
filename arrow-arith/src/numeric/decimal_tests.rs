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

use super::*;

fn assert_matches_array<T: DecimalType>(
    op: DecimalOp,
    left: (Option<T::Native>, u8, i8),
    right: (Option<T::Native>, u8, i8),
) {
    let expected = (|| {
        let left_array =
            PrimitiveArray::<T>::from_iter([left.0]).with_precision_and_scale(left.1, left.2)?;
        let right_array =
            PrimitiveArray::<T>::from_iter([right.0]).with_precision_and_scale(right.1, right.2)?;
        let left = Scalar::new(left_array);
        let right = Scalar::new(right_array);
        let result = match op {
            DecimalOp::Add => add(&left, &right),
            DecimalOp::Sub => sub(&left, &right),
            DecimalOp::Mul => mul(&left, &right),
            DecimalOp::Div => div(&left, &right),
            DecimalOp::Rem => rem(&left, &right),
        }?;
        let result = result.as_primitive::<T>();
        Ok::<_, ArrowError>((
            (!result.is_null(0)).then(|| result.value(0)),
            result.precision(),
            result.scale(),
        ))
    })();
    assert_eq!(
        decimal_scalar::<T>(op, left, right).map_err(|e| e.to_string()),
        expected.map_err(|e| e.to_string()),
        "{op:?}: {left:?}, {right:?}",
    );
}

fn check_width<T: DecimalType>() {
    let value = |v| Some(T::Native::usize_as(v));
    for op in [
        DecimalOp::Add,
        DecimalOp::Sub,
        DecimalOp::Mul,
        DecimalOp::Div,
        DecimalOp::Rem,
    ] {
        for (left, right) in [
            ((value(123), 8, 2), (value(7), 7, 2)),
            ((value(123), 8, 2), (value(7), 7, 1)),
            ((value(10), 3, -2), (value(3), 2, -1)),
            ((value(3), 1, i8::MIN), (value(2), 1, i8::MIN)),
            ((None, 8, 2), (value(0), 7, 1)),
            ((value(0), 8, 2), (None, 7, 1)),
        ] {
            assert_matches_array::<T>(op, left, right);
        }
    }
}

#[test]
fn scalar_matches_all_decimal_widths() {
    check_width::<Decimal32Type>();
    check_width::<Decimal64Type>();
    check_width::<Decimal128Type>();
    check_width::<Decimal256Type>();
}

#[test]
fn scalar_preserves_decimal_errors() {
    for (op, left, right) in [
        (DecimalOp::Add, (Some(i128::MAX), 38, 0), (Some(1), 1, 0)),
        (DecimalOp::Sub, (Some(i128::MIN), 38, 0), (Some(1), 1, 0)),
        (DecimalOp::Mul, (Some(i128::MAX), 38, 0), (Some(2), 1, 0)),
        (DecimalOp::Div, (Some(1), 1, 0), (Some(0), 1, 0)),
        (DecimalOp::Rem, (Some(1), 1, 0), (Some(0), 1, 0)),
        (DecimalOp::Rem, (Some(i128::MIN), 38, 0), (Some(-1), 1, 0)),
        (DecimalOp::Mul, (None, 38, 38), (None, 1, 1)),
        (DecimalOp::Div, (None, 38, -2), (Some(1), 38, 38)),
        (DecimalOp::Mul, (Some(1), 0, 0), (Some(0), 1, 0)),
        (DecimalOp::Div, (Some(1), 1, 0), (None, 1, 2)),
    ] {
        assert_matches_array::<Decimal128Type>(op, left, right);
    }
}

#[test]
fn null_coefficients_preserve_scale_preparation_errors() {
    for (op, left, right) in [
        (DecimalOp::Add, (None, 1, -10), (Some(1), 1, 0)),
        (DecimalOp::Sub, (None, 1, -10), (Some(1), 1, 0)),
        (DecimalOp::Div, (None, 9, 9), (Some(1), 1, -10)),
    ] {
        assert!(matches!(
            decimal_scalar::<Decimal32Type>(op, left, right),
            Err(ArrowError::ArithmeticOverflow(_))
        ));
    }

    // Remainder uses wrapping scale multipliers, so preparing 10^10 does not fail.
    assert_eq!(
        decimal_scalar::<Decimal32Type>(DecimalOp::Rem, (None, 1, -10), (Some(1), 1, 0)).unwrap(),
        (None, 1, 0),
    );
}

#[test]
fn coefficient_errors_precede_output_metadata_validation() {
    // These input scales produce an invalid result precision of zero.
    // Division by zero must still be reported before validating that precision.
    assert!(matches!(
        decimal_scalar::<Decimal32Type>(DecimalOp::Div, (Some(1), 1, 0), (Some(0), 1, -5)),
        Err(ArrowError::DivideByZero)
    ));

    // Output metadata is validated after successful or skipped coefficient arithmetic.
    for value in [Some(1), None] {
        assert_eq!(
            decimal_scalar::<Decimal32Type>(DecimalOp::Div, (value, 1, 0), (Some(1), 1, -5))
                .unwrap_err()
                .to_string(),
            "Invalid argument error: precision cannot be 0, has to be between [1, 9]",
        );
    }
}

fn check_wide_division<T: DecimalType>() {
    let unit = T::Native::usize_as(10)
        .pow_checked(u32::from(T::MAX_PRECISION) - 1)
        .unwrap();
    let left = unit.mul_checked(T::Native::usize_as(3)).unwrap();
    let right = unit.mul_checked(T::Native::usize_as(6)).unwrap();
    let expected = unit.mul_checked(T::Native::usize_as(5)).unwrap();
    for (left, expected) in [
        (left, expected),
        (left.neg_checked().unwrap(), expected.neg_checked().unwrap()),
    ] {
        let left = (Some(left), T::MAX_PRECISION, T::MAX_SCALE - 1);
        let right = (Some(right), T::MAX_PRECISION, T::MAX_SCALE - 1);
        assert_matches_array::<T>(DecimalOp::Div, left, right);
        assert_eq!(
            decimal_scalar::<T>(DecimalOp::Div, left, right).unwrap(),
            (Some(expected), T::MAX_PRECISION, T::MAX_SCALE),
        );
    }
}

#[test]
fn scalar_division_preserves_wide_intermediate_fallback() {
    check_wide_division::<Decimal32Type>();
    check_wide_division::<Decimal64Type>();
    check_wide_division::<Decimal128Type>();
    check_wide_division::<Decimal256Type>();
}
