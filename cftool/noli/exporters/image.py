from ..schema import RenderType
from ..schema import RenderParams
from ...geometry import Matrix2D


DEFAULT_RENDER_TYPE = RenderType.COVER


def get_img_transform(w: float, h: float, render_params: RenderParams) -> Matrix2D:
    img_transform = Matrix2D.scale_matrix(w, h)
    crop_transform = render_params.cropFields
    if crop_transform is None:
        return img_transform
    return img_transform @ crop_transform


def get_img_render_transform(
    bbox: Matrix2D,
    img_w: float,
    img_h: float,
    render_params: RenderParams,
) -> Matrix2D:
    img_transform = get_img_transform(img_w, img_h, render_params)
    render_transform = img_transform.inverse
    render_type = render_params.renderType or DEFAULT_RENDER_TYPE
    if render_type == RenderType.FILL:
        return render_transform
    w, h = bbox.wh
    h = max(1.0e-8, abs(h))
    area_wh_ratio = w / h
    img_wh_ratio = img_w / img_h
    ratio = img_wh_ratio / area_wh_ratio
    is_fit = render_type == RenderType.FIT
    if (ratio >= 1.0 and not is_fit) or (ratio < 1.0 and is_fit):
        w_scale, h_scale = ratio, 1.0
    else:
        w_scale, h_scale = 1.0, 1.0 / ratio
    scale_center = Matrix2D.identical().center
    render_transform = render_transform.scale(w_scale, h_scale, scale_center)
    return render_transform


__all__ = [
    "DEFAULT_RENDER_TYPE",
    "get_img_transform",
    "get_img_render_transform",
]
