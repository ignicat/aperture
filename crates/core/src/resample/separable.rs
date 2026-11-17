use crate::arena::{CompiledResamplePlan, PlanCache, ResamplePlanKey};
use crate::color::{WorkingSpace, global_luts};
use crate::error::{EngineError, EngineResult};
use crate::pixel::{PackedImage, PixelFormat, PlanarF32};
use crate::pool::ThreadPool;
use crate::resample::kernels::{ResampleFilter, build_axis_plan};
use crate::resample::simd::resample_separable_planar_f32;
use fast_image_resize::images::Image;
use fast_image_resize::{PixelComponentMapper, PixelType, ResizeAlg, ResizeOptions, Resizer, create_srgb_mapper};

pub fn resample_separable(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    filter: ResampleFilter,
    antiring: f32,
    plan_cache: &PlanCache,
    pool: &ThreadPool,
) -> EngineResult<PlanarF32> {
    if src.width == dst_w && src.height == dst_h {
        return Ok(src.clone());
    }
    if dst_w == 0 || dst_h == 0 {
        return Err(EngineError::InvalidDimensions {
            width: dst_w,
            height: dst_h,
        });
    }

    if antiring <= 0.0
        && src.working_space == WorkingSpace::LinearSrgb
        && matches!(
            filter,
            ResampleFilter::Lanczos3
                | ResampleFilter::Mitchell
                | ResampleFilter::CatmullRom
                | ResampleFilter::Box
                | ResampleFilter::Nearest
        )
        && src.width <= 16384
        && src.height <= 16384
        && dst_w <= 16384
        && dst_h <= 16384
        && let Ok(fast) = try_fast_image_resize_planar(src, dst_w, dst_h, filter)
    {
        return Ok(fast);
    }

    let key = ResamplePlanKey {
        src_w: src.width,
        src_h: src.height,
        dst_w,
        dst_h,
        kernel_id: filter.id(),
        quantized_param: 0,
    };
    let plan = plan_cache.get_or_build(key, || CompiledResamplePlan {
        horiz: build_axis_plan(src.width, dst_w, filter),
        vert: build_axis_plan(src.height, dst_h, filter),
    });

    resample_separable_planar_f32(src, &plan.horiz, &plan.vert, antiring, pool)
}

fn try_fast_image_resize_planar(
    src: &PlanarF32,
    dst_w: u32,
    dst_h: u32,
    filter: ResampleFilter,
) -> EngineResult<PlanarF32> {
    let luts = global_luts();
    let fmt = if src.has_alpha {
        PixelFormat::Rgba16
    } else {
        PixelFormat::Rgb16
    };
    let packed_src = src.to_packed(fmt, luts)?;
    let pt = if src.has_alpha {
        PixelType::U16x4
    } else {
        PixelType::U16x3
    };
    let fir_src = Image::from_vec_u8(src.width, src.height, packed_src.data, pt)
        .map_err(|e| EngineError::EncodeFailed(e.to_string()))?;
    let mut fir_dst = Image::new(dst_w, dst_h, pt);
    let alg = match filter {
        ResampleFilter::Nearest => ResizeAlg::Nearest,
        ResampleFilter::Box => ResizeAlg::Convolution(fast_image_resize::FilterType::Box),
        ResampleFilter::Mitchell => ResizeAlg::Convolution(fast_image_resize::FilterType::Mitchell),
        ResampleFilter::CatmullRom => {
            ResizeAlg::Convolution(fast_image_resize::FilterType::CatmullRom)
        }
        _ => ResizeAlg::Convolution(fast_image_resize::FilterType::Lanczos3),
    };
    let opts = ResizeOptions::new().resize_alg(alg).use_alpha(src.has_alpha);
    let mut resizer = Resizer::new();
    let srgb_mapper: PixelComponentMapper = create_srgb_mapper();
    let mut linear_src = Image::new(src.width, src.height, pt);
    srgb_mapper
        .forward_map(&fir_src, &mut linear_src)
        .map_err(|e| EngineError::ColorTransform(e.to_string()))?;
    let mut linear_dst = Image::new(dst_w, dst_h, pt);
    resizer
        .resize(&linear_src, &mut linear_dst, Some(&opts))
        .map_err(|e| EngineError::ColorTransform(e.to_string()))?;
    srgb_mapper
        .backward_map(&linear_dst, &mut fir_dst)
        .map_err(|e| EngineError::ColorTransform(e.to_string()))?;
    let packed_dst = PackedImage::new(
        dst_w,
        dst_h,
        fmt,
        crate::color::ColorSpace::Srgb,
        fir_dst.into_vec(),
    )?;
    PlanarF32::from_packed(&packed_dst, WorkingSpace::LinearSrgb, luts)
}
