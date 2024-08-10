use opencv::core::Size;
use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionDataConditional, ActionSequence, ActionWhile,
            TupleSecond,
        },
        basic::DelayAction,
        extra::{AlwaysTrue, CountFalse, CountTrue, IsSome, OutputType, Terminal},
        movement::{
            AdjustType, ClampX, ConstYaw, Descend, LinearYawFromX, MultiplyX, NoAdjust,
            OffsetToPose, SetX, Stability2Adjust, Stability2Movement, Stability2Pos, StripY,
            ZeroMovement,
        },
        vision::{DetectTarget, ExtractPosition, MidPoint, Norm, Vision},
    },
    vision::{
        path::{Path, Yuv},
        Offset2D,
    },
};

use super::{
    action::ActionExec,
    action_context::{GetControlBoard, GetFrontCamMat, GetMainElectronicsBoard},
};

pub fn octagon_path_model() -> Path {
    Path::new(
        (Yuv {
            y: 0,
            u: 127,
            v: 127,
        })..=(Yuv {
            y: 255,
            u: 255,
            v: 255,
        }),
        5.0..=800.0,
        4,
        Size::from((400, 300)),
        3,
    )
}

pub fn octagon<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    const FULL_SPEED_Y: f32 = 0.7;
    const FULL_SPEED_X: f32 = 0.0;
    const FULL_SPEED_PITCH: f32 = -45.0 / 4.0;
    const DEPTH: f32 = -0.75;

    const INIT_X: f32 = 0.5;
    const INIT_Y: f32 = 0.0;
    const INIT_TIME: f32 = 3.0;

    const BLIND_TIME: f32 = 3.0;

    const X_CLAMP: f32 = 0.3;

    const FALSE_COUNT: u32 = 3;
    const ADJUST_COUNT: u32 = 2;

    act_nest!(
        ActionSequence::new,
        ActionWhile::new(act_nest!(
            ActionSequence::new,
            act_nest!(
                ActionChain::new,
                NoAdjust::<Stability2Adjust>::new(),
                ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(80.0)),
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)
                ),
                OutputType::<()>::new(),
            ),
            DelayAction::new(1.0),
            ActionChain::<bool, _, _>::new(AlwaysTrue::default(), CountTrue::new(ADJUST_COUNT)),
        ),),
        DelayAction::new(2.0),
        ActionChain::new(
            Stability2Movement::new(
                context,
                Stability2Pos::new(INIT_X, INIT_Y, 0.0, 0.0, None, DEPTH)
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(INIT_TIME),
        act_nest!(
            ActionChain::new,
            Stability2Movement::new(
                context,
                Stability2Pos::new(
                    FULL_SPEED_X,
                    FULL_SPEED_Y,
                    FULL_SPEED_PITCH,
                    0.0,
                    None,
                    DEPTH
                )
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(BLIND_TIME),
        ActionWhile::new(ActionSequence::new(
            act_nest!(
                ActionChain::new,
                Stability2Movement::new(
                    context,
                    Stability2Pos::new(
                        FULL_SPEED_X,
                        FULL_SPEED_Y,
                        FULL_SPEED_PITCH,
                        0.0,
                        None,
                        DEPTH
                    )
                ),
                OutputType::<()>::new(),
            ),
            act_nest!(
                ActionChain::new,
                Vision::<Con, Path, f64>::new(context, octagon_path_model()),
                IsSome::default(),
                CountTrue::new(1)
            )
        )),
        ActionWhile::new(act_nest!(
            ActionChain::new,
            Vision::<Con, Path, f64>::new(context, octagon_path_model()),
            TupleSecond::<_, bool>::new(ActionConcurrent::new(
                ActionSequence::new(
                    act_nest!(
                        ActionChain::new,
                        ActionDataConditional::new(
                            DetectTarget::new(true),
                            act_nest!(
                                ActionChain::new,
                                Norm::new(Path::default()),
                                ExtractPosition::new(),
                                MidPoint::new(),
                                OffsetToPose::<Offset2D<f64>>::default(),
                                LinearYawFromX::<Stability2Adjust>::new(7.0),
                                MultiplyX::new(0.5),
                                ClampX::<Stability2Adjust>::new(X_CLAMP),
                            ),
                            act_nest!(
                                ActionSequence::new,
                                Terminal::new(),
                                SetX::<Stability2Adjust>::new(AdjustType::Replace(FULL_SPEED_X)),
                            )
                        ),
                        StripY::<Stability2Adjust>::new(),
                        Stability2Movement::new(
                            context,
                            Stability2Pos::new(FULL_SPEED_X, FULL_SPEED_Y, 0.0, 0.0, None, DEPTH)
                        ),
                        OutputType::<()>::new(),
                    ),
                    AlwaysTrue::new()
                ),
                ActionChain::new(IsSome::default(), CountFalse::new(FALSE_COUNT))
            )),
        ),),
        ZeroMovement::new(context, DEPTH),
        OutputType::<()>::new()
    )
}

#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;

    use opencv::{
        core::Vector,
        imgcodecs::{imread, imwrite, IMREAD_COLOR},
    };
    use rayon::iter::{ParallelBridge, ParallelIterator};

    use crate::{
        logln,
        vision::{Draw, VisualDetection, VisualDetector},
    };

    use super::*;

    #[test]
    fn distance_detect() {
        let mut model = octagon_path_model();
        let image = imread(
            "tests/vision/resources/new_octagon_images/distance.png",
            IMREAD_COLOR,
        )
        .unwrap();

        let output: Vec<_> = <Path as VisualDetector<f64>>::detect(&mut model, &image)
            .unwrap()
            .into_iter()
            .filter(|x| *x.class())
            .collect();
        logln!("{:#?}", output);
        assert_eq!(output.len(), 0);

        create_dir_all("tests/vision/output/octagon_images").unwrap();
        imwrite(
            "tests/vision/output/octagon_images/distance_detect.png",
            &model.image(),
            &Vector::default(),
        )
        .unwrap();
    }

    #[test]
    fn close_detect() {
        let mut model = octagon_path_model();
        let image = imread(
            "tests/vision/resources/new_octagon_images/close.png",
            IMREAD_COLOR,
        )
        .unwrap();

        let output: Vec<_> = <Path as VisualDetector<f64>>::detect(&mut model, &image)
            .unwrap()
            .into_iter()
            .filter(|x| *x.class())
            .collect();
        logln!("{:#?}", output);

        assert_eq!(output.len(), 1);

        let mut shrunk_image = model.image().clone();
        output.iter().for_each(|result| {
            <VisualDetection<_, _> as Draw>::draw(result, &mut shrunk_image).unwrap()
        });

        create_dir_all("tests/vision/output/octagon_images").unwrap();
        imwrite(
            "tests/vision/output/octagon_images/close_detect.png",
            &shrunk_image,
            &Vector::default(),
        )
        .unwrap();
    }

    #[test]
    fn full_video_detects() {
        const ENTERS_VISION: usize = 0;
        const LEAVES_VISION: usize = 0;

        std::fs::read_dir("tests/vision/resources/new_octagon_images/roll_pics")
            .unwrap()
            .enumerate()
            .par_bridge()
            .for_each(|(idx, f)| {
                let mut model = octagon_path_model();
                let image = imread(f.unwrap().path().to_str().unwrap(), IMREAD_COLOR).unwrap();

                let output: Vec<_> = <Path as VisualDetector<f64>>::detect(&mut model, &image)
                    .unwrap()
                    .into_iter()
                    .filter(|x| *x.class())
                    .collect();
                logln!("{:#?}", output);

                /*
                if idx > ENTERS_VISION && idx < LEAVES_VISION {
                    assert!(!output.is_empty());
                } else {
                    assert_eq!(output.len(), 0);
                }
                */

                let mut shrunk_image = model.image().clone();
                output.iter().for_each(|result| {
                    <VisualDetection<_, _> as Draw>::draw(result, &mut shrunk_image).unwrap()
                });

                create_dir_all("tests/vision/output/octagon_images/roll_pics").unwrap();
                imwrite(
                    &format!(
                        "tests/vision/output/octagon_images/roll_pics/{:#03}.png",
                        idx
                    ),
                    &shrunk_image,
                    &Vector::default(),
                )
                .unwrap();
            })
    }
}
