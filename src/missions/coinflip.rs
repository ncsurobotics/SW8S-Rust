pub fn coin_flip_mission< 
    Con: Send
        + Sync
        + GetControlBoard<WriteHalf<SerialStream>>
        + GetMainElectronicsBoard
        + GetFrontCamMat
        + Unpin,
>(
    context: &'static Con,
) -> impl ActionExec<()> + '_ {
    let mut rng = rand::thread_rng();
    let coin_result = rng.gen_bool(0.5);
    let orientation = if coin_result { 90.0 } else { 180.0 };

    act_nest!(
        ActionSequence::new,

        // Initial orientation based on the coin flip  
        ActionChain::new(
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, 0.0, 0.0, 0.0, Some(orientation), DEPTH)
            ),
            OutputType::<()>::new(),
        ),
        DelayAction::new(2.0), // Delay to stabilize orientation  

        // Search for the gate using computer vision
        ActionChain::new(
            VisionNorm::new(context, Gate::default()),  
            OutputType::<Result<Vec<VisualDetection<Target, Offset2D<f64>>>>>::new(),
        ),

        // Detect the gate with YOLO-based classification
        ActionChain::new(
            DetectTarget::new(Target::LargeGate),
            OutputType::<Option<Vec<VisualDetection<Target, Offset2D<f64>>>>>::new(),
        ),

        // Extract position, align before moving forward  
        ActionWhile::new(
            CountTrue::new(MAX_ADJUSTMENT_STEPS),  
            MAX_ADJUSTMENT_STEPS,  

            ActionSequence::new(  
                ActionChain::new(
                    ExtractPosition::<Target, Offset2D<f64>>::new(),  
                    OutputType::<Option<Vec<Offset2D<f64>>>>::new(),
                ),
                ActionChain::new(
                    OffsetToPose::<Offset2D<f64>>::new(Offset2D::default()),  
                    Stability2Adjust::new(context, AdjustType::default()),   
                    OutputType::<()>::new(),
                ),
            ),
        ),

        // Ensure final orientation before moving 
        ActionChain::new(
            Stability2Movement::new(
                context,
                Stability2Pos::new(0.0, 0.0, 0.0, 0.0, None, DEPTH)  
            ),
            OutputType::<()>::new(),
        ),

        DelayAction::new(1.0), // Small delay before moving forward  

        // Move forward toward the detected gate
        ActionChain::new(
            Stability2Movement::new(
                context, 
                Stability2Pos::new(0.0, STANDARD_DISTANCE, 0.0, 0.0, None, DEPTH)
            ),
            OutputType::<()>::new(),
        ),

        OutputType::<()>::new() 
    )
}
