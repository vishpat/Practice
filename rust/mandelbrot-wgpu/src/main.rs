use pollster::block_on;
use wgpu::util::DeviceExt;
use ndarray::Array2;

const WORKGROUP_SIZE: u64 = 128;
const WIDTH: usize = (WORKGROUP_SIZE * 20) as usize;
const HEIGHT: usize = (WORKGROUP_SIZE * 20) as usize;
const SIZE: wgpu::BufferAddress = (WIDTH * HEIGHT) as wgpu::BufferAddress;

#[repr(C)]
#[derive(Debug, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub x: f32,
    pub y: f32,
    pub x_range: f32,
    pub y_range: f32,
    pub max_iter: u32,
}

async fn gpu_device_queue() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    (device, queue)
}

async fn compute_shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MandelBrot Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot.wgsl").into()),
    })
}

async fn cpu_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CPU Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

async fn gpu_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU Buffer"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

async fn run() {
    let (device, queue) = gpu_device_queue().await;

    let params = Params {
        width: WIDTH as u32,
        height: HEIGHT as u32,
        x: -0.65,
        y: 0.0,
        x_range: 3.4,
        y_range: 3.4,
        max_iter: 1000,
    };
    let gpu_param_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("GPU Parameter Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let size = (SIZE as usize * std::mem::size_of::<u32>()) as u64;
    let cpu_buf = cpu_buffer(&device, size).await;
    let gpu_buf = gpu_buffer(&device, size).await;

    let cs_module = compute_shader(&device).await;
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MandelBrot Compute Pipeline"),
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buf.as_entire_binding(),
        }],
    });

    let param_group_layout = compute_pipeline.get_bind_group_layout(1);
    let param_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &param_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_param_buf.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Command Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.set_bind_group(1, &param_bind_group, &[]);
        cpass.insert_debug_marker("MandelBrot Compute Pass");
        cpass.dispatch_workgroups((SIZE / WORKGROUP_SIZE) as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&gpu_buf, 0, &cpu_buf, 0, size);

    queue.submit(Some(encoder.finish()));
    let buffer_slice = cpu_buf.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if let Ok(Ok(())) = receiver.recv_async().await {
        let data = buffer_slice.get_mapped_range();
        let _result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        let pixels = Array2::from_shape_vec((HEIGHT, WIDTH), _result).unwrap();
        let img = image::ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
            let pixel = pixels[[y as usize, x as usize]];
            image::Rgb([(pixel >> 16) as u8, (pixel >> 8) as u8, pixel as u8])
        });
        img.save("mandelbrot.png").unwrap();
        drop(data);
        cpu_buf.unmap();
    } else {
        panic!("failed to run compute on gpu!")
    }
}

fn main() {
    env_logger::init();
    block_on(run());
}
