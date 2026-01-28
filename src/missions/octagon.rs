use tokio::io::WriteHalf;
use tokio_serial::SerialStream;

use crate::{
    act_nest,
    config::{octagon::Config, ColorProfile},
    missions::{
        action::{
            ActionChain, ActionConcurrent, ActionDataConditional, ActionSequence, ActionWhile,
            RaceAction, TupleSecond,
        },
        basic::DelayAction,
        extra::{
            AlwaysBetterFalse, AlwaysBetterTrue, AlwaysTrue, CountFalse, CountTrue, OutputType,
            Terminal, ToVec,
        },
        movement::{
            AdjustType, ClampX, ConstYaw, LinearYawFromX, NoAdjust, OffsetToPose, SetX,
            Stability2Adjust, Stability2Movement, Stability2Pos, StripY, ZeroMovement,
        },
        vision::{DetectTarget, ExtractPosition, MidPoint, Norm, Vision},
    },
    vision::{octagon::Octagon, Offset2D},
    POOL_YAW_SIGN,
};

use super::{
    action::ActionExec,
    action_context::{FrontCamIO, GetControlBoard, GetMainElectronicsBoard},
};

pub fn octagon_path_model() -> Octagon {
    Octagon::default()
}

pub fn octagon<
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + FrontCamIO
        + Unpin,
>(
    context: &'static Con,
    config: &Config,
    color_profile: &ColorProfile,
) -> impl ActionExec<()> + 'static {
    let _ = config;
    let _ = color_profile;
    const FULL_SPEED_Y: f32 = 0.7;
    const FULL_SPEED_X: f32 = 0.0;
    const FULL_SPEED_PITCH: f32 = -45.0 / 4.0;
    const DEPTH: f32 = -0.75;

    const INIT_X: f32 = 0.0;
    const INIT_Y: f32 = 0.0;
    const INIT_TIME: f32 = 3.0;

    const BLIND_TIME: f32 = 3.0;

    const X_CLAMP: f32 = 0.3;

    const FALSE_COUNT: u32 = 3;
    const ADJUST_COUNT: u32 = 2;

    const OCTAGON_SPIN: f32 = 50.0 * POOL_YAW_SIGN;

    const MISSION_END_TIME: f32 = ((INIT_TIME + BLIND_TIME) * 2.0) + 13.0 + 6.0;

    RaceAction::new(
        act_nest!(
            ActionSequence::new,
            ActionWhile::new(act_nest!(
                ActionSequence::new,
                act_nest!(
                    ActionChain::new,
                    NoAdjust::<Stability2Adjust>::new(),
                    ConstYaw::<Stability2Adjust>::new(AdjustType::Adjust(OCTAGON_SPIN)),
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
                    Vision::<Con, Octagon, f64>::new(context, octagon_path_model()),
                    TupleSecond::new(ActionConcurrent::new(
                        act_nest!(
                            ActionChain::new,
                            ToVec::new(),
                            Norm::new(Octagon::default()),
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::<Offset2D<f64>>::default(),
                            LinearYawFromX::<Stability2Adjust>::new(7.0),
                            ClampX::<Stability2Adjust>::new(X_CLAMP),
                            StripY::<Stability2Adjust>::new(),
                            ActionChain::new(
                                Stability2Movement::new(
                                    context,
                                    Stability2Pos::new(
                                        FULL_SPEED_X,
                                        FULL_SPEED_Y,
                                        0.0,
                                        0.0,
                                        None,
                                        DEPTH
                                    )
                                ),
                                OutputType::<()>::new(),
                            ),
                        ),
                        ActionChain::new(DetectTarget::new(true), CountTrue::new(1)),
                    ))
                )
            )),
            ActionWhile::new(act_nest!(
                ActionChain::new,
                Vision::<Con, Octagon, f64>::new(context, octagon_path_model()),
                ActionDataConditional::new(
                    DetectTarget::new(true),
                    ActionSequence::new(
                        act_nest!(
                            ActionChain::new,
                            Norm::new(Octagon::default()),
                            ExtractPosition::new(),
                            MidPoint::new(),
                            OffsetToPose::<Offset2D<f64>>::default(),
                            LinearYawFromX::<Stability2Adjust>::new(7.0),
                            ClampX::<Stability2Adjust>::new(X_CLAMP),
                            StripY::<Stability2Adjust>::new(),
                            ActionChain::new(
                                Stability2Movement::new(
                                    context,
                                    Stability2Pos::new(
                                        FULL_SPEED_X,
                                        FULL_SPEED_Y,
                                        0.0,
                                        0.0,
                                        None,
                                        DEPTH
                                    )
                                ),
                                OutputType::<()>::new(),
                            ),
                        ),
                        AlwaysBetterTrue::new(),
                    ),
                    ActionSequence::new(
                        act_nest!(
                            ActionSequence::new,
                            Terminal::new(),
                            SetX::<Stability2Adjust>::new(AdjustType::Replace(FULL_SPEED_X)),
                            StripY::<Stability2Adjust>::new(),
                            ActionChain::new(
                                Stability2Movement::new(
                                    context,
                                    Stability2Pos::new(
                                        FULL_SPEED_X,
                                        FULL_SPEED_Y,
                                        0.0,
                                        0.0,
                                        None,
                                        DEPTH
                                    )
                                ),
                                OutputType::<()>::new(),
                            ),
                        ),
                        AlwaysBetterFalse::new(),
                    ),
                ),
                CountFalse::new(FALSE_COUNT)
            ),),
            ZeroMovement::new(context, DEPTH),
            OutputType::<()>::new()
        ),
        DelayAction::new(MISSION_END_TIME),
    )
}

#[cfg(test)]
mod tests {
    use std::fs::{create_dir_all, remove_dir_all};

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

        let output: Vec<_> = <Octagon as VisualDetector<f64>>::detect(&mut model, &image)
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

        let output: Vec<_> = <Octagon as VisualDetector<f64>>::detect(&mut model, &image)
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

                let output: Vec<_> = <Octagon as VisualDetector<f64>>::detect(&mut model, &image)
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

    #[test]
    fn real_video_detects() {
        const ENTERS_VISION: usize = 0;
        const LEAVES_VISION: usize = 0;

        let _ = remove_dir_all("tests/vision/output/octagon_images/octagon_real_run");

        std::fs::read_dir("tests/vision/resources/octagon_real_run/")
            .unwrap()
            .enumerate()
            .par_bridge()
            .for_each(|(idx, f)| {
                let mut model = octagon_path_model();
                let image = imread(f.unwrap().path().to_str().unwrap(), IMREAD_COLOR).unwrap();

                let output_prefilter: Vec<_> =
                    <Octagon as VisualDetector<f64>>::detect(&mut model, &image).unwrap();

                let output: Vec<_> = output_prefilter
                    .clone()
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
                output_prefilter.iter().for_each(|result| {
                    <VisualDetection<_, _> as Draw>::draw(result, &mut shrunk_image).unwrap()
                });

                create_dir_all("tests/vision/output/octagon_images/octagon_real_run").unwrap();
                imwrite(
                    &format!(
                        "tests/vision/output/octagon_images/octagon_real_run/{:#03}.png",
                        idx
                    ),
                    &shrunk_image,
                    &Vector::default(),
                )
                .unwrap();
            })
    }
}
