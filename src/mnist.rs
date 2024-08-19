use std::{
    fmt::{Display, Formatter},
    fs::File,
    io::BufReader,
    ops::{Div, Mul},
};

use byteorder::{BigEndian, ReadBytesExt};
use ndarray::Array1;
use num_traits::{Float, FromPrimitive, ToPrimitive};

pub struct Image<T> {
    pub image: Array1<T>,
    pub classification: u8,
}

impl<T> Image<T> {
    pub fn load(dataset: &str) -> Result<Vec<Image<T>>, std::io::Error>
    where
        T: FromPrimitive + Div<Output = T>,
    {
        Ok({
            let label_file = File::open(format!("data/{}-labels-idx1-ubyte", dataset)).unwrap();
            let image_file = File::open(format!("data/{}-images-idx3-ubyte", dataset)).unwrap();
            let mut label_buffer = BufReader::new(label_file);
            let mut image_buffer = BufReader::new(image_file);
            let magic_number = label_buffer.read_i32::<BigEndian>()?;
            assert!(magic_number == 2049);
            let magic_number = image_buffer.read_i32::<BigEndian>()?;
            assert!(magic_number == 2051);
            let num_labels = label_buffer.read_i32::<BigEndian>()?;
            let num_images = image_buffer.read_i32::<BigEndian>()?;
            assert!(num_labels == num_images);
            let num_rows = image_buffer.read_i32::<BigEndian>()?;
            let num_columns = image_buffer.read_i32::<BigEndian>()?;
            (0..num_images)
                .map(|_| {
                    let vec = (0..num_rows * num_columns)
                        .map(|_| {
                            Ok(T::from_u8(image_buffer.read_u8()?).unwrap()
                                / T::from_u8(255).unwrap())
                        })
                        .collect::<Result<_, std::io::Error>>()?;
                    let image = Array1::from_vec(vec);
                    Ok(Image {
                        image,
                        classification: label_buffer.read_u8()?,
                    })
                })
                .collect::<Result<_, std::io::Error>>()?
        })
    }
}

impl<T> Display for Image<T>
where
    T: FromPrimitive + ToPrimitive + Mul<Output = T> + Float,
{
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        let shades = [" ", "░", "▒", "▓", "█"];
        writeln!(
            f,
            "{}{}{}{}{}",
            "┌",
            "─".repeat(13),
            self.classification,
            "─".repeat(14),
            "┐"
        )?;
        for (i, pixel) in self.image.iter().enumerate() {
            if i % 28 == 0 {
                write!(f, "│")?;
            }
            let shade = (*pixel * T::from_usize(shades.len()).unwrap())
                .floor()
                .to_usize()
                .unwrap()
                .min(shades.len() - 1);
            write!(f, "{}", shades[shade])?;

            if i % 28 == 27 {
                writeln!(f, "│")?;
            }
        }
        writeln!(f, "{}{}{}", "└", "─".repeat(28), "┘")?;
        Ok(())
    }
}
