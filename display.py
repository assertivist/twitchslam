
import sdl2
import sdl2.ext

sdl2.ext.init()

class Display(object):
    def __init__(self, W, H):
          
        self.W, self.H = W, H
        self.window = sdl2.ext.Window("twitch SLAM", size=(W,H))
        self.window.show() 

    def paint(self, img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = img.swapaxes(0, 1)
        self.window.refresh()
