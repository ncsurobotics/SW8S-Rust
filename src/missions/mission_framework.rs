trait State {
    fn on_enter(&self);
    fn on_periodic(&self);
    fn on_exit(&self);
    fn next_state(&self) -> Option<Box<dyn State>>; // return trait object
}


struct EmptyState;

impl State for EmptyState {
    fn on_enter(&self) {
        stub!(); 
    }
    fn on_periodic(&self) {
        stub!();
    }
    fn on_exit(&self) {
        stub!();
    }
    fn next_state(&self) -> Option<Box<dyn State>> {
        None
    }

}


trait Mission {
    type InitialState: State;

    fn new(initial_state: Self::InitialState) -> Self;
    fn current_state(&self) -> Option<&Box<dyn State>>;
    fn set_current_state(&mut self, state: Box<dyn State>);

    // Default implementation
    fn run(&mut self) {
        self.current_state().on_enter();
        self.current_state().on_periodic();
        self.current_state().on_exit();
        let next = self.current_state().next_state();// todo, make this good.  
        match next {
            Some(state) => {
                self.set_current_state(state);
            },
            None => {
                // we do not have a real next state, what do we do? 
                stub!(); 
            }
        }
    }
}