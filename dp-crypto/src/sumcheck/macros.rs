macro_rules! entered_span {
    ($first:expr $(, $($fields:tt)*)?) => {
        tracing_span!($first, $($($fields)*)?).entered()
    }
}

macro_rules! tracing_span {
    ($first:expr $(, $($fields:tt)*)?) => {
        tracing::span!(tracing::Level::TRACE, $first, $($($fields)*)?)
    }
}

macro_rules! exit_span {
    ($first:expr $(,)*) => {
        $first.exit()
    };
}

#[allow(unused_imports)]
pub(crate) use entered_span;
#[allow(unused_imports)]
pub(crate) use exit_span;
