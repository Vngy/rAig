import moderngl
import numpy as np

from raig.core.params import ParamFrame
from raig.core.rig import Rig
from raig.render.deform import deform_rig

CHROMA_GREEN = (0.0, 1.0, 0.0, 1.0)

_VERT = """
#version 330
uniform vec2 canvas;
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    vec2 ndc = vec2(in_pos.x / canvas.x * 2.0 - 1.0,
                    1.0 - in_pos.y / canvas.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_FRAG = """
#version 330
uniform sampler2D tex;
uniform sampler2D mask_tex;
// mode 0 = normal, 1 = masked by mask_tex, 2 = write alpha into red channel
uniform int mode;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 c = texture(tex, v_uv);  // premultiplied
    if (mode == 2) { f_color = vec4(c.a, 0.0, 0.0, 1.0); return; }
    if (mode == 1) {
        float m = texelFetch(mask_tex, ivec2(gl_FragCoord.xy), 0).r;
        c *= m;
    }
    f_color = c;
}
"""


def _premultiply(texture: np.ndarray) -> np.ndarray:
    t = texture.astype(np.float32) / 255.0
    t[..., :3] *= t[..., 3:4]
    return (t * 255.0 + 0.5).astype(np.uint8)


class _GpuLayer:
    def __init__(self, ctx, program, layer):
        self.layer = layer
        tex = _premultiply(layer.texture)
        self.texture = ctx.texture(
            (tex.shape[1], tex.shape[0]), 4, tex.tobytes()
        )
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.pos_vbo = ctx.buffer(
            layer.vertices.astype("f4").tobytes(), dynamic=True
        )
        uv_vbo = ctx.buffer(layer.uvs.astype("f4").tobytes())
        ibo = ctx.buffer(layer.triangles.astype("i4").tobytes())
        self.vao = ctx.vertex_array(
            program,
            [(self.pos_vbo, "2f4", "in_pos"), (uv_vbo, "2f4", "in_uv")],
            index_buffer=ibo,
        )


class RigRenderer:
    def __init__(self, ctx: moderngl.Context, rig: Rig):
        self.ctx = ctx
        self.rig = rig
        self.program = ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        self.program["canvas"].value = tuple(map(float, rig.canvas_size))
        self.program["tex"].value = 0
        self.program["mask_tex"].value = 1
        self.layers = {l.layer_name: _GpuLayer(ctx, self.program, l) for l in rig.layers}
        self._mask_fbo = None
        self._mask_tex = None

    def _mask_fbo_for(self, fbo):
        if self._mask_fbo is None or self._mask_fbo.size != fbo.size:
            tex = self.ctx.texture(fbo.size, 1, dtype="f1")
            self._mask_fbo = self.ctx.framebuffer(color_attachments=[tex])
            self._mask_tex = tex
        return self._mask_fbo

    def render_frame(
        self,
        frame: ParamFrame,
        fbo: moderngl.Framebuffer,
        background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        deformed = deform_rig(self.rig, frame)
        for gpu in self.layers.values():
            gpu.pos_vbo.write(deformed[gpu.layer.layer_name].astype("f4").tobytes())

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA)
        fbo.use()
        fbo.clear(*background)
        mask_fbo = self._mask_fbo_for(fbo)
        # Keep unit 1 bound even for unmasked layers: a sampler uniform
        # pointing at an unbound unit is undefined on some GL drivers.
        self._mask_tex.use(1)

        for gpu in self.layers.values():
            layer = gpu.layer
            if layer.clip_to and layer.clip_to in self.layers:
                mask_gpu = self.layers[layer.clip_to]
                mask_fbo.use()
                mask_fbo.clear(0.0, 0.0, 0.0, 0.0)
                self.ctx.disable(moderngl.BLEND)
                self.program["mode"].value = 2
                mask_gpu.texture.use(0)
                mask_gpu.vao.render()
                self.ctx.enable(moderngl.BLEND)
                fbo.use()
                self.program["mode"].value = 1
                self._mask_tex.use(1)
            else:
                self.program["mode"].value = 0
            gpu.texture.use(0)
            gpu.vao.render()


def render_to_image(
    rig: Rig,
    frame: ParamFrame,
    size: tuple[int, int] | None = None,
    background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    size = size or rig.canvas_size
    ctx = moderngl.create_context(standalone=True)
    try:
        fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 4)])
        RigRenderer(ctx, rig).render_frame(frame, fbo, background)
        data = np.frombuffer(fbo.read(components=4), dtype=np.uint8)
        img = data.reshape(size[1], size[0], 4)
        return img[::-1].copy()  # GL reads bottom-up; return origin top-left
    finally:
        ctx.release()
