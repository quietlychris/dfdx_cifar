#[inline]
pub fn max_index(v: &Vec<f64>) -> usize {
    let mut index = 0;
    let mut max = v[0];
    for i in 1..v.len() {
        if v[i] > max {
            index = i;
            max = v[i];
        }
    }
    index
}
