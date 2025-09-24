import { useCallback, useEffect, useRef, useState } from 'react'
import { Accordion, Box, Button, Container, Flex, Grid, Group, Loader, NumberFormatter, Paper, Text, TextInput, Title } from '@mantine/core'

interface GenerationStats {
  current_fps: number
  average_fps: number
  min_fps: number
  max_fps: number
  total_generations: number
  timing_breakdown?: TimingBreakdown
  memory_stats?: MemoryStats
}

interface TimingBreakdown {
  total_time_ms: number
  unet_time_ms: number
  vae_encode_time_ms: number
  vae_decode_time_ms: number
  memory_transfer_time_ms: number
  websocket_time_ms: number
  unet_percentage: number
  vae_encode_percentage: number
  vae_decode_percentage: number
  memory_transfer_percentage: number
  websocket_percentage: number
  bottleneck: string
}

interface MemoryStats {
  current_memory_mb: number
  average_memory_mb: number
  peak_memory_mb: number
  memory_efficiency: number
}

interface ConfigParameter {
  value: any
  description: string
  type: string
}

interface ConfigSection {
  title: string
  description: string
  parameters: Record<string, ConfigParameter>
}

interface Configuration {
  [sectionName: string]: ConfigSection
}

export const App = () => {
  const [inputPrompt, setInputPrompt] = useState('a photo of a cat')
  const [generatedImage, setGeneratedImage] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [currentFps, setCurrentFps] = useState<number>(0)
  const [lastInferenceTime, setLastInferenceTime] = useState<number>(0)
  const [isConnected, setIsConnected] = useState(false)
  const [timingBreakdown, setTimingBreakdown] = useState<TimingBreakdown | null>(null)
  const [memoryStats, setMemoryStats] = useState<MemoryStats | null>(null)
  const [stats, setStats] = useState<GenerationStats>({
    current_fps: 0,
    average_fps: 0,
    min_fps: 0,
    max_fps: 0,
    total_generations: 0
  })
  const [config, setConfig] = useState<Configuration | null>(null)

  const wsRef = useRef<WebSocket | null>(null)

  const connectWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws`

    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data)

      if (message.type === 'image' || message.type === 'benchmark_image') {
        const data = message.data
        const imageUrl = `data:image/jpeg;base64,${data.base64_image}`

        setGeneratedImage(imageUrl)
        setCurrentFps(data.fps)
        setLastInferenceTime(data.inference_time_ms)
        setStats(data.stats)

        // Update detailed timing and memory data if available
        if (data.timing_breakdown) {
          setTimingBreakdown(data.timing_breakdown)
        }
        if (data.memory_stats) {
          setMemoryStats(data.memory_stats)
        }

        setIsLoading(false)
      } else if (message.type === 'stats') {
        setStats(message.data)
        if (message.data.timing_breakdown) {
          setTimingBreakdown(message.data.timing_breakdown)
        }
        if (message.data.memory_stats) {
          setMemoryStats(message.data.memory_stats)
        }
      } else if (message.type === 'config') {
        setConfig(message.data)
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      setIsConnected(false)
      setIsGenerating(false)
      // Attempt to reconnect after 1 second
      setTimeout(connectWebSocket, 1000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }

    wsRef.current = ws
  }, [])

  const generateImage = useCallback(() => {
    if (!wsRef.current || !inputPrompt.trim()) return

    setIsLoading(true)
    wsRef.current.send(JSON.stringify({
      type: 'generate',
      prompt: inputPrompt
    }))
  }, [inputPrompt])

  const startContinuousGeneration = useCallback(() => {
    if (!wsRef.current || !inputPrompt.trim()) return

    setIsGenerating(true)
    wsRef.current.send(JSON.stringify({
      type: 'start_benchmark',
      prompt: inputPrompt
    }))
  }, [inputPrompt])

  const stopContinuousGeneration = useCallback(() => {
    if (!wsRef.current) return

    setIsGenerating(false)
    wsRef.current.send(JSON.stringify({
      type: 'stop_benchmark'
    }))
  }, [])

  // Request configuration on WebSocket connection
  const requestConfig = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'get_config' }))
    }
  }, [])

  // WebSocket connection effect
  useEffect(() => {
    connectWebSocket()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connectWebSocket])

  // Request config when connected
  useEffect(() => {
    if (isConnected) {
      // Small delay to ensure WebSocket is fully ready
      setTimeout(requestConfig, 100)
    }
  }, [isConnected, requestConfig])

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return (
    <Box bg="#1a1b1e" mih="100vh" w="100vw" p="xl">
      <Container size="lg">
        <Flex justify="space-between" align="center" mb="xl">
          <Title order={1} c="white">
            txt2imgBench - Hyper-SD MPS Performance
          </Title>
          <Flex align="center" gap="sm">
            <Box
              w={12}
              h={12}
              bg={isConnected ? 'green' : 'red'}
              style={{ borderRadius: '50%' }}
            />
            <Text c={isConnected ? 'green' : 'red'} size="sm">
              {isConnected ? 'WebSocket Connected' : 'Disconnected'}
            </Text>
          </Flex>
        </Flex>

        <Grid>
          {/* Left column - Controls */}
          <Grid.Col span={6}>
            <Paper p="md" bg="#25262b">
              <Title order={3} c="white" mb="md">Controls</Title>

              <TextInput
                label="Prompt"
                placeholder="Enter your prompt"
                value={inputPrompt}
                onChange={(event) => setInputPrompt(event.currentTarget.value)}
                mb="md"
                styles={{
                  label: { color: 'white' },
                  input: { backgroundColor: '#373a40', color: 'white', borderColor: '#5c5f66' }
                }}
              />

              <Group mb="md">
                <Button
                  onClick={generateImage}
                  disabled={!isConnected || isLoading || isGenerating}
                  variant="filled"
                  color="blue"
                >
                  Generate Single Image
                </Button>

                {!isGenerating ? (
                  <Button
                    onClick={startContinuousGeneration}
                    disabled={!isConnected || isLoading}
                    variant="filled"
                    color="green"
                  >
                    Start Benchmark
                  </Button>
                ) : (
                  <Button
                    onClick={stopContinuousGeneration}
                    variant="filled"
                    color="red"
                  >
                    Stop Benchmark
                  </Button>
                )}
              </Group>

              {isLoading && (
                <Flex align="center" gap="sm">
                  <Loader size="sm" />
                  <Text c="white">Generating...</Text>
                </Flex>
              )}
            </Paper>

            {/* Performance Stats */}
            <Paper p="md" bg="#25262b" mt="md">
              <Title order={3} c="white" mb="md">Performance Stats</Title>

              <Grid>
                <Grid.Col span={6}>
                  <Text c="dimmed" size="sm">Current FPS</Text>
                  <Text c="white" fw={700} size="lg">
                    <NumberFormatter value={currentFps} decimalScale={2} />
                  </Text>
                </Grid.Col>

                <Grid.Col span={6}>
                  <Text c="dimmed" size="sm">Last Inference</Text>
                  <Text c="white" fw={700} size="lg">
                    <NumberFormatter value={lastInferenceTime} decimalScale={1} />ms
                  </Text>
                </Grid.Col>

                <Grid.Col span={6}>
                  <Text c="dimmed" size="sm">Average FPS</Text>
                  <Text c="white" fw={700}>
                    <NumberFormatter value={stats.average_fps} decimalScale={2} />
                  </Text>
                </Grid.Col>

                <Grid.Col span={6}>
                  <Text c="dimmed" size="sm">Total Images</Text>
                  <Text c="white" fw={700}>
                    {stats.total_generations}
                  </Text>
                </Grid.Col>

                <Grid.Col span={6}>
                  <Text c="dimmed" size="sm">Min FPS</Text>
                  <Text c="white" fw={700}>
                    <NumberFormatter value={stats.min_fps} decimalScale={2} />
                  </Text>
                </Grid.Col>

                <Grid.Col span={6}>
                  <Text c="dimmed" size="sm">Max FPS</Text>
                  <Text c="white" fw={700}>
                    <NumberFormatter value={stats.max_fps} decimalScale={2} />
                  </Text>
                </Grid.Col>
              </Grid>
            </Paper>

            {/* Performance Analysis */}
            {timingBreakdown && (
              <Paper p="md" bg="#25262b" mt="md">
                <Title order={3} c="white" mb="md">
                  Performance Analysis
                  {timingBreakdown.bottleneck && (
                    <Text size="sm" c="orange" ml="sm" style={{display: 'inline'}}>
                      Bottleneck: {timingBreakdown.bottleneck}
                    </Text>
                  )}
                </Title>

                <Grid>
                  <Grid.Col span={12}>
                    <Text c="dimmed" size="sm" mb="xs">Component Timing (ms)</Text>
                  </Grid.Col>

                  <Grid.Col span={6}>
                    <Box>
                      <Text c="white" size="sm">UNet Inference</Text>
                      <Flex justify="space-between">
                        <Text c="white" fw={700}>
                          <NumberFormatter value={timingBreakdown.unet_time_ms} decimalScale={1} />ms
                        </Text>
                        <Text c="blue" size="sm">
                          {timingBreakdown.unet_percentage.toFixed(1)}%
                        </Text>
                      </Flex>
                    </Box>
                  </Grid.Col>

                  <Grid.Col span={6}>
                    <Box>
                      <Text c="white" size="sm">VAE Decode</Text>
                      <Flex justify="space-between">
                        <Text c="white" fw={700}>
                          <NumberFormatter value={timingBreakdown.vae_decode_time_ms} decimalScale={1} />ms
                        </Text>
                        <Text c="green" size="sm">
                          {timingBreakdown.vae_decode_percentage.toFixed(1)}%
                        </Text>
                      </Flex>
                    </Box>
                  </Grid.Col>

                  <Grid.Col span={6}>
                    <Box>
                      <Text c="white" size="sm">Memory Transfer</Text>
                      <Flex justify="space-between">
                        <Text c="white" fw={700}>
                          <NumberFormatter value={timingBreakdown.memory_transfer_time_ms} decimalScale={2} />ms
                        </Text>
                        <Text c="yellow" size="sm">
                          {timingBreakdown.memory_transfer_percentage.toFixed(1)}%
                        </Text>
                      </Flex>
                    </Box>
                  </Grid.Col>

                  <Grid.Col span={6}>
                    <Box>
                      <Text c="white" size="sm">WebSocket</Text>
                      <Flex justify="space-between">
                        <Text c="white" fw={700}>
                          <NumberFormatter value={timingBreakdown.websocket_time_ms} decimalScale={2} />ms
                        </Text>
                        <Text c="purple" size="sm">
                          {timingBreakdown.websocket_percentage.toFixed(1)}%
                        </Text>
                      </Flex>
                    </Box>
                  </Grid.Col>
                </Grid>
              </Paper>
            )}

            {/* Memory Analysis */}
            {memoryStats && (
              <Paper p="md" bg="#25262b" mt="md">
                <Title order={3} c="white" mb="md">Memory Usage (MPS)</Title>

                <Grid>
                  <Grid.Col span={6}>
                    <Text c="dimmed" size="sm">Current</Text>
                    <Text c="white" fw={700}>
                      <NumberFormatter value={memoryStats.current_memory_mb} decimalScale={1} />MB
                    </Text>
                  </Grid.Col>

                  <Grid.Col span={6}>
                    <Text c="dimmed" size="sm">Peak</Text>
                    <Text c="white" fw={700}>
                      <NumberFormatter value={memoryStats.peak_memory_mb} decimalScale={1} />MB
                    </Text>
                  </Grid.Col>

                  <Grid.Col span={6}>
                    <Text c="dimmed" size="sm">Average</Text>
                    <Text c="white" fw={700}>
                      <NumberFormatter value={memoryStats.average_memory_mb} decimalScale={1} />MB
                    </Text>
                  </Grid.Col>

                  <Grid.Col span={6}>
                    <Text c="dimmed" size="sm">Efficiency</Text>
                    <Text c={memoryStats.memory_efficiency > 80 ? 'green' : 'orange'} fw={700}>
                      {memoryStats.memory_efficiency.toFixed(1)}%
                    </Text>
                  </Grid.Col>
                </Grid>
              </Paper>
            )}

            {/* Configuration Display */}
            {config && (
              <Paper p="md" bg="#25262b" mt="md">
                <Title order={3} c="white" mb="md">Configuration Parameters</Title>

                <Accordion
                  variant="separated"
                  styles={{
                    item: { backgroundColor: '#373a40', border: '1px solid #5c5f66' },
                    control: { color: 'white' },
                    label: { color: 'white' },
                    content: { color: 'white' }
                  }}
                >
                  {Object.entries(config).map(([sectionKey, section]) => (
                    <Accordion.Item key={sectionKey} value={sectionKey}>
                      <Accordion.Control>
                        <Flex justify="space-between" align="center">
                          <Box>
                            <Text fw={600}>{section.title}</Text>
                            <Text size="sm" c="dimmed">{section.description}</Text>
                          </Box>
                        </Flex>
                      </Accordion.Control>
                      <Accordion.Panel>
                        <Grid>
                          {Object.entries(section.parameters).map(([paramKey, param]) => (
                            <Grid.Col key={paramKey} span={6}>
                              <Box>
                                <Text size="sm" fw={500}>{paramKey}</Text>
                                <Text size="xs" c="dimmed" mb="xs">{param.description}</Text>
                                <Text
                                  c={param.type === 'boolean' ? (param.value ? 'green' : 'orange') : 'white'}
                                  fw={600}
                                  style={{
                                    fontFamily: 'monospace',
                                    backgroundColor: '#2d2d2d',
                                    padding: '4px 8px',
                                    borderRadius: '4px',
                                    display: 'inline-block'
                                  }}
                                >
                                  {Array.isArray(param.value)
                                    ? `[${param.value.join(', ')}]`
                                    : typeof param.value === 'boolean'
                                    ? param.value.toString()
                                    : param.value}
                                </Text>
                              </Box>
                            </Grid.Col>
                          ))}
                        </Grid>
                      </Accordion.Panel>
                    </Accordion.Item>
                  ))}
                </Accordion>
              </Paper>
            )}
          </Grid.Col>

          {/* Right column - Generated Image */}
          <Grid.Col span={6}>
            <Paper p="md" bg="#25262b" h="100%">
              <Title order={3} c="white" mb="md">Generated Image</Title>

              <Box
                style={{
                  width: '100%',
                  height: '400px',
                  backgroundColor: '#373a40',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden'
                }}
              >
                {generatedImage ? (
                  <img
                    src={generatedImage}
                    alt="Generated"
                    style={{
                      maxWidth: '100%',
                      maxHeight: '100%',
                      objectFit: 'contain'
                    }}
                  />
                ) : (
                  <Text c="dimmed" ta="center">
                    {isLoading ? 'Generating...' : 'No image generated yet'}
                  </Text>
                )}
              </Box>

              {isGenerating && (
                <Text c="green" ta="center" mt="sm" fw={700}>
                  🔥 Benchmarking in progress...
                </Text>
              )}
            </Paper>
          </Grid.Col>
        </Grid>
      </Container>
    </Box>
  )
}