use rust_decimal::Decimal;

use crate::error::VerifierError;

pub const MAX_EXPRESSION_LEN: usize = 4_096;
pub const MAX_RECURSION_DEPTH: usize = 64;
pub const MAX_EXPONENT: u64 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Number(Decimal),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LParen,
    RParen,
}

impl Token {
    const fn describe(self) -> &'static str {
        match self {
            Self::Number(_) => "number",
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Star => "*",
            Self::Slash => "/",
            Self::Percent => "%",
            Self::Caret => "^",
            Self::LParen => "(",
            Self::RParen => ")",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PositionedToken {
    token: Token,
    position: usize,
}

/// Avalia uma expressão matemática usando aritmética decimal 100% exata (sem ponto flutuante).
///
/// Suporta `+`, `-`, `*`, `/`, `%`, `^` (expoentes inteiros), unários `+`/`-` e parênteses.
///
/// # Errors
///
/// Retorna [`VerifierError`] em casos de:
/// - sintaxe inválida, expressão vazia ou parênteses não fechados;
/// - divisão ou módulo por zero;
/// - *overflow* decimal ou expoente não inteiro;
/// - limites de segurança excedidos (tamanho, profundidade de recursão ou magnitude de expoente).
pub fn evaluate_decimal(input: &str) -> Result<Decimal, VerifierError> {
    if input.len() > MAX_EXPRESSION_LEN {
        return Err(VerifierError::ExpressionTooLong {
            max: MAX_EXPRESSION_LEN,
        });
    }

    let tokens = tokenize(input)?;
    if tokens.is_empty() {
        return Err(VerifierError::EmptyExpression);
    }

    let mut parser = Parser::new(&tokens);
    let value = parser.parse_add_sub(0)?;

    if let Some(extra) = parser.peek() {
        return Err(VerifierError::UnexpectedToken {
            token: extra.token.describe(),
            position: extra.position,
        });
    }

    Ok(value.normalize())
}

fn tokenize(input: &str) -> Result<Vec<PositionedToken>, VerifierError> {
    let mut tokens = Vec::with_capacity(input.len() / 2 + 1);
    let mut chars = input.char_indices().peekable();

    while let Some((pos, ch)) = chars.next() {
        if ch.is_ascii_whitespace() {
            continue;
        }

        let single = match ch {
            '+' => Some(Token::Plus),
            '-' => Some(Token::Minus),
            '*' => Some(Token::Star),
            '/' => Some(Token::Slash),
            '%' => Some(Token::Percent),
            '^' => Some(Token::Caret),
            '(' => Some(Token::LParen),
            ')' => Some(Token::RParen),
            _ => None,
        };

        if let Some(token) = single {
            tokens.push(PositionedToken {
                token,
                position: pos,
            });
            continue;
        }

        if ch.is_ascii_digit() || ch == '.' {
            let mut end = pos.saturating_add(ch.len_utf8());
            let mut has_dot = ch == '.';
            let mut has_digit = ch.is_ascii_digit();

            while let Some(&(next_pos, next_ch)) = chars.peek() {
                if next_ch.is_ascii_digit() {
                    has_digit = true;
                    end = next_pos.saturating_add(next_ch.len_utf8());
                    chars.next();
                } else if next_ch == '.' {
                    if has_dot {
                        end = next_pos.saturating_add(next_ch.len_utf8());
                        chars.next();
                        let literal = input[pos..end].to_owned();
                        return Err(VerifierError::InvalidNumber {
                            literal,
                            position: pos,
                        });
                    }
                    has_dot = true;
                    end = next_pos.saturating_add(next_ch.len_utf8());
                    chars.next();
                } else {
                    break;
                }
            }

            let raw = &input[pos..end];
            if !has_digit {
                return Err(VerifierError::InvalidNumber {
                    literal: raw.to_owned(),
                    position: pos,
                });
            }

            let parsed = parse_decimal_literal(raw, pos)?;
            tokens.push(PositionedToken {
                token: Token::Number(parsed),
                position: pos,
            });
            continue;
        }

        return Err(VerifierError::UnexpectedCharacter { ch, position: pos });
    }

    Ok(tokens)
}

fn parse_decimal_literal(raw: &str, position: usize) -> Result<Decimal, VerifierError> {
    let owned;
    let normalized = if raw.starts_with('.') {
        owned = format!("0{raw}");
        owned.as_str()
    } else if raw.ends_with('.') {
        owned = format!("{raw}0");
        owned.as_str()
    } else {
        raw
    };

    Decimal::from_str_exact(normalized).map_err(|err| match err {
        rust_decimal::Error::ExceedsMaximumPossibleValue
        | rust_decimal::Error::LessThanMinimumPossibleValue => VerifierError::Overflow,
        _ => VerifierError::InvalidNumber {
            literal: raw.to_owned(),
            position,
        },
    })
}

struct Parser<'a> {
    tokens: &'a [PositionedToken],
    cursor: usize,
}

impl<'a> Parser<'a> {
    const fn new(tokens: &'a [PositionedToken]) -> Self {
        Self { tokens, cursor: 0 }
    }

    fn peek(&self) -> Option<PositionedToken> {
        self.tokens.get(self.cursor).copied()
    }

    fn advance(&mut self) {
        self.cursor = self.cursor.saturating_add(1);
    }

    fn parse_add_sub(&mut self, depth: usize) -> Result<Decimal, VerifierError> {
        let mut lhs = self.parse_mul_div(depth)?;

        while let Some(op) = self.peek().map(|t| t.token) {
            match op {
                Token::Plus => {
                    self.advance();
                    let rhs = self.parse_mul_div(depth)?;
                    lhs = lhs.checked_add(rhs).ok_or(VerifierError::Overflow)?;
                }
                Token::Minus => {
                    self.advance();
                    let rhs = self.parse_mul_div(depth)?;
                    lhs = lhs.checked_sub(rhs).ok_or(VerifierError::Overflow)?;
                }
                _ => break,
            }
        }

        Ok(lhs)
    }

    fn parse_mul_div(&mut self, depth: usize) -> Result<Decimal, VerifierError> {
        let mut lhs = self.parse_unary(depth)?;

        while let Some(op) = self.peek().map(|t| t.token) {
            match op {
                Token::Star => {
                    self.advance();
                    let rhs = self.parse_unary(depth)?;
                    lhs = lhs.checked_mul(rhs).ok_or(VerifierError::Overflow)?;
                }
                Token::Slash => {
                    self.advance();
                    let rhs = self.parse_unary(depth)?;
                    if rhs.is_zero() {
                        return Err(VerifierError::DivisionByZero);
                    }
                    lhs = lhs.checked_div(rhs).ok_or(VerifierError::Overflow)?;
                }
                Token::Percent => {
                    self.advance();
                    let rhs = self.parse_unary(depth)?;
                    if rhs.is_zero() {
                        return Err(VerifierError::DivisionByZero);
                    }
                    lhs = lhs.checked_rem(rhs).ok_or(VerifierError::Overflow)?;
                }
                _ => break,
            }
        }

        Ok(lhs)
    }

    fn parse_unary(&mut self, depth: usize) -> Result<Decimal, VerifierError> {
        match self.peek().map(|t| t.token) {
            Some(Token::Plus) => {
                self.advance();
                self.parse_unary(next_depth(depth)?)
            }
            Some(Token::Minus) => {
                self.advance();
                let value = self.parse_unary(next_depth(depth)?)?;
                // Decimal usa representação sinal-magnitude; a negação nunca sofre overflow.
                Ok(-value)
            }
            _ => self.parse_power(depth),
        }
    }

    fn parse_power(&mut self, depth: usize) -> Result<Decimal, VerifierError> {
        let base = self.parse_primary(depth)?;

        if matches!(self.peek().map(|t| t.token), Some(Token::Caret)) {
            self.advance();
            let exponent = self.parse_unary(next_depth(depth)?)?;
            return checked_pow_decimal(base, exponent);
        }

        Ok(base)
    }

    fn parse_primary(&mut self, depth: usize) -> Result<Decimal, VerifierError> {
        match self.peek() {
            Some(PositionedToken {
                token: Token::Number(value),
                ..
            }) => {
                self.advance();
                Ok(value)
            }
            Some(PositionedToken {
                token: Token::LParen,
                position: open_pos,
            }) => {
                self.advance();
                let inner = self.parse_add_sub(next_depth(depth)?)?;
                match self.peek() {
                    Some(PositionedToken {
                        token: Token::RParen,
                        ..
                    }) => {
                        self.advance();
                        Ok(inner)
                    }
                    _ => Err(VerifierError::UnclosedParenthesis { position: open_pos }),
                }
            }
            Some(other) => Err(VerifierError::UnexpectedToken {
                token: other.token.describe(),
                position: other.position,
            }),
            None => Err(VerifierError::UnexpectedEndOfInput),
        }
    }
}

fn next_depth(depth: usize) -> Result<usize, VerifierError> {
    let next = depth
        .checked_add(1)
        .ok_or(VerifierError::MaxDepthExceeded {
            max: MAX_RECURSION_DEPTH,
        })?;
    if next > MAX_RECURSION_DEPTH {
        return Err(VerifierError::MaxDepthExceeded {
            max: MAX_RECURSION_DEPTH,
        });
    }
    Ok(next)
}

fn checked_pow_decimal(base: Decimal, exponent: Decimal) -> Result<Decimal, VerifierError> {
    if !exponent.fract().is_zero() {
        return Err(VerifierError::NonIntegerExponent);
    }

    let exp_i64 = i64::try_from(exponent).map_err(|_| VerifierError::ExponentOutOfRange {
        max: MAX_EXPONENT,
    })?;
    let abs_exp = exp_i64.unsigned_abs();
    if abs_exp > MAX_EXPONENT {
        return Err(VerifierError::ExponentOutOfRange { max: MAX_EXPONENT });
    }

    if exp_i64 == 0 {
        return Ok(Decimal::ONE);
    }
    if exp_i64 < 0 && base.is_zero() {
        return Err(VerifierError::DivisionByZero);
    }

    let mut result = Decimal::ONE;
    let mut factor = base;
    let mut remaining = abs_exp;

    while remaining > 0 {
        if remaining & 1 == 1 {
            result = result.checked_mul(factor).ok_or(VerifierError::Overflow)?;
        }
        remaining >>= 1;
        if remaining > 0 {
            factor = factor.checked_mul(factor).ok_or(VerifierError::Overflow)?;
        }
    }

    if exp_i64 < 0 {
        Decimal::ONE
            .checked_div(result)
            .ok_or(VerifierError::Overflow)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(s: &str) -> Result<Decimal, Box<dyn std::error::Error>> {
        Ok(Decimal::from_str_exact(s)?)
    }

    #[test]
    fn avoids_ieee754_floating_point_error() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(evaluate_decimal("0.1 + 0.2")?, dec("0.3")?);
        assert_eq!(
            evaluate_decimal("(199.90 * 3 - 49.70) * 1.15")?,
            dec("632.5")?
        );
        Ok(())
    }

    #[test]
    fn respects_precedence_parentheses_and_associativity()
    -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(evaluate_decimal("2 + 3 * 4")?, dec("14")?);
        assert_eq!(evaluate_decimal("(2 + 3) * 4")?, dec("20")?);
        assert_eq!(evaluate_decimal("10 - 3 - 2")?, dec("5")?);
        assert_eq!(evaluate_decimal("10 % 3")?, dec("1")?);
        assert_eq!(evaluate_decimal("-2 ^ 2")?, dec("-4")?);
        assert_eq!(evaluate_decimal("(-2) ^ 2")?, dec("4")?);
        assert_eq!(evaluate_decimal("2 ^ 3 ^ 2")?, dec("512")?);
        assert_eq!(evaluate_decimal("2 ^ -2")?, dec("0.25")?);
        assert_eq!(evaluate_decimal(".5 + .25")?, dec("0.75")?);
        Ok(())
    }

    #[test]
    fn catches_division_by_zero_and_overflow() {
        assert_eq!(
            evaluate_decimal("10 / (5 - 5)"),
            Err(VerifierError::DivisionByZero)
        );
        assert_eq!(
            evaluate_decimal("10 % 0"),
            Err(VerifierError::DivisionByZero)
        );
        assert_eq!(
            evaluate_decimal("0 ^ -1"),
            Err(VerifierError::DivisionByZero)
        );
        assert_eq!(
            evaluate_decimal(&format!("{} + 1", Decimal::MAX)),
            Err(VerifierError::Overflow)
        );
        assert_eq!(
            evaluate_decimal("10 ^ 50"),
            Err(VerifierError::Overflow)
        );
    }

    #[test]
    fn catches_syntax_and_resource_limits() {
        assert_eq!(evaluate_decimal("   "), Err(VerifierError::EmptyExpression));
        assert_eq!(
            evaluate_decimal("4 ^ 0.5"),
            Err(VerifierError::NonIntegerExponent)
        );
        assert_eq!(
            evaluate_decimal("2 ^ 5000"),
            Err(VerifierError::ExponentOutOfRange { max: MAX_EXPONENT })
        );
        assert!(matches!(
            evaluate_decimal("(1 + 2"),
            Err(VerifierError::UnclosedParenthesis { position: 0 })
        ));
        assert!(matches!(
            evaluate_decimal("1.2.3"),
            Err(VerifierError::InvalidNumber { .. })
        ));

        let deep = format!("{}1{}", "(".repeat(100), ")".repeat(100));
        assert_eq!(
            evaluate_decimal(&deep),
            Err(VerifierError::MaxDepthExceeded {
                max: MAX_RECURSION_DEPTH,
            })
        );
    }
}
