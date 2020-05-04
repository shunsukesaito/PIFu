_glut_window = None
_context_inited = None

def initialize_GL_context(width=512, height=512, egl=False):
    '''
    default context uses GLUT
    '''
    if not egl:
        import OpenGL.GLUT as GLUT      
        display_mode = GLUT.GLUT_DOUBLE | GLUT.GLUT_RGB | GLUT.GLUT_DEPTH
        global _glut_window
        if _glut_window is None:
            GLUT.glutInit()
            GLUT.glutInitDisplayMode(display_mode)
            GLUT.glutInitWindowSize(width, height)
            GLUT.glutInitWindowPosition(0, 0)
            _glut_window = GLUT.glutCreateWindow("My Render.")
    else:
        from .glcontext import create_opengl_context
        global _context_inited
        if _context_inited is None:
            create_opengl_context((width, height))
            _context_inited = True

